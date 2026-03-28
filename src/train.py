"""
Fine-tuning LoRA do Qwen3.5 com PEFT + TRL.
Ambiente agnóstico: CUDA (QLoRA 4-bit) | MPS Apple Silicon (bf16) | CPU (fp32).

Uso:
    uv run python -m src.train
    # ou via Makefile:
    make train
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_ckpt
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from trl import SFTConfig, SFTTrainer

from src.config.settings import settings

TRAIN_FILE   = Path("data/processed/train.jsonl")
VALID_FILE   = Path("data/processed/valid.jsonl")
LORA_HF_DIR  = Path("models/lora-hf")

# ─── Limites de memória por ambiente ─────────────────────────────────────────
# CUDA: conservador para GPUs de 8 GB VRAM (modelo 4-bit ~4.5 GB, ~2 GB para gradientes)
CUDA_MAX_MEMORY = "6500MiB"


# ─── Detecção de dispositivo ──────────────────────────────────────────────────

def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> Dataset:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


# ─── Modelo ───────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(device: str):
    model_id = settings.model_hf_id
    load_kwargs: dict = {}

    print(f"  → Modelo HF: {model_id}")

    if device == "cuda":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            load_kwargs.update({
                "quantization_config": bnb,
                "device_map": {"": 0},
            })
            print("  → Modo: QLoRA 4-bit (CUDA, tudo na GPU 0)")
        except ImportError:
            load_kwargs.update({
                "torch_dtype": torch.float16,
                "device_map": {"": 0},
            })
            print("  → Modo: float16 (CUDA, tudo na GPU 0)")

    elif device == "mps":
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs.update({"quantization_config": bnb, "device_map": {"": "mps"}})
        print("  → Modo: QLoRA 4-bit (Apple Silicon MPS, conservador para 8 GB)")

    else:
        torch.set_num_threads(max(1, settings.cpu_threads - 2))
        os.environ["OMP_NUM_THREADS"] = str(max(1, settings.cpu_threads - 2))
        load_kwargs.update({"torch_dtype": torch.float32, "low_cpu_mem_usage": True})
        print(f"  → Modo: float32 (CPU, threads: {max(1, settings.cpu_threads - 2)})")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ─── Trainer com CE fatiada ───────────────────────────────────────────────────

class _ChunkedLMHeadTrainer(SFTTrainer):
    """
    Evita OOM em GPUs ≤8 GB com vocabulário grande (Qwen3.5: 248k tokens).

    O problema: `logits.float()` no ForCausalLMLoss do transformers cria um
    tensor fp32 de ~154 MB de uma vez — impossível quando a GPU tem <100 MB livres
    após carregar o modelo 9B em 4-bit.

    Solução: calcula lm_head + cross-entropy em fatias de _CHUNK tokens com
    gradient checkpointing por fatia. O tensor de logits completo nunca existe
    em memória; pico cai de ~154 MB para ~30 MB por fatia.
    """
    _CHUNK = 32

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")

        # Estrutura do modelo PEFT: PeftModelForCausalLM → LoraModel → Qwen3_5ForCausalLM
        qwen = model.base_model.model         # Qwen3_5ForCausalLM
        transformer = qwen.model              # Qwen3_5Model (sem lm_head)
        lm_head = qwen.lm_head                # Linear com LoRA aplicado

        # Forward pelo transformer até os hidden states (sem lm_head / sem loss)
        valid_inputs = {
            k: v for k, v in inputs.items()
            if k in {"input_ids", "attention_mask", "position_ids", "inputs_embeds"}
        }
        hidden_out = transformer(**valid_inputs, use_cache=False, return_dict=True)
        hidden = hidden_out.last_hidden_state  # [B, T, D] — já normalizados pelo Qwen3_5Model

        T = hidden.shape[1]
        n_valid = int((labels[:, 1:T] != -100).sum())
        total_loss = hidden.new_zeros((), dtype=torch.float32)

        def _ce_chunk(h_chunk, lbl):
            """Calcula CE de uma fatia sem reter logits entre forward e backward."""
            logits = lm_head(h_chunk).float()
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                lbl,
                ignore_index=-100,
                reduction="sum",
            )

        for start in range(0, T - 1, self._CHUNK):
            end = min(start + self._CHUNK, T - 1)
            lbl = labels[:, start + 1 : end + 1].contiguous().view(-1)
            if (lbl != -100).any():
                h = hidden[:, start:end].contiguous()
                total_loss = total_loss + grad_ckpt(_ce_chunk, h, lbl, use_reentrant=False)

        loss = total_loss / max(n_valid, 1)
        out = CausalLMOutputWithPast(loss=loss)
        return (loss, out) if return_outputs else loss


# ─── Treinamento ──────────────────────────────────────────────────────────────

def main() -> None:
    device = detect_device()
    print(f"\n→ Dispositivo: {device.upper()}")

    LORA_HF_DIR.mkdir(parents=True, exist_ok=True)

    print("→ Carregando modelo e tokenizer...")
    model, tokenizer = load_model_and_tokenizer(device)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("→ Carregando datasets...")
    train_dataset = load_jsonl(TRAIN_FILE)
    valid_dataset = load_jsonl(VALID_FILE) if VALID_FILE.exists() else None
    print(f"  Treino: {len(train_dataset)} | Validação: {len(valid_dataset) if valid_dataset else 0}")

    use_bf16 = device in ("cuda", "mps")
    grad_accum = 8 if device == "mps" else 4
    gc_kwargs = {"use_reentrant": False} if device in ("cuda", "mps") else {}

    training_args = SFTConfig(
        output_dir=str(LORA_HF_DIR),
        max_steps=settings.train_iters,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=gc_kwargs,
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        dataloader_pin_memory=(device == "cuda"),
        dataloader_num_workers=0,
        eval_strategy="steps" if valid_dataset else "no",
        eval_steps=100 if valid_dataset else None,
        dataset_text_field="text",
        max_length=256,
        packing=False,
    )

    trainer = _ChunkedLMHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Parâmetros treináveis: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    print(f"\n→ Iniciando fine-tuning ({settings.train_iters} steps)...")
    trainer.train()

    print(f"\n→ Salvando adaptador LoRA em {LORA_HF_DIR}...")
    trainer.save_model(str(LORA_HF_DIR))
    tokenizer.save_pretrained(str(LORA_HF_DIR))

    print(f"\n✓ Adaptador HF salvo em: {LORA_HF_DIR}")
    print("  Próximo passo: make export-lora")


if __name__ == "__main__":
    main()
