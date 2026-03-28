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
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
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
        # device_map={"": 0} força todo o modelo na GPU 0.
        # bitsandbytes 4-bit não suporta offload para CPU/disco — usar "auto"
        # com max_memory faz o accelerate tentar offload e causa ValueError.
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
        # QLoRA 4-bit limita o modelo a ~4.5 GB de memória unificada.
        # Em Macs com 8 GB, o sistema reserva ~1-2 GB → sobram ~2-3 GB para
        # ativações e estados do otimizador LoRA. max_length=512 é obrigatório.
        # set_per_process_memory_fraction não é aplicado durante o carregamento
        # pois o bitsandbytes carrega parcialmente em bf16, causando OOM falso.
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
        # CPU: limita threads para não saturar todos os núcleos
        torch.set_num_threads(max(1, settings.cpu_threads - 2))
        os.environ["OMP_NUM_THREADS"] = str(max(1, settings.cpu_threads - 2))
        load_kwargs.update({"torch_dtype": torch.float32, "low_cpu_mem_usage": True})
        print(f"  → Modo: float32 (CPU, threads: {max(1, settings.cpu_threads - 2)})")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


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

    # Em MPS com 8 GB: acumula mais passos para compensar o batch pequeno
    # e usar use_reentrant=False reduz o pico de memória no backward.
    grad_accum = 8 if device == "mps" else 4
    gc_kwargs = {"use_reentrant": False} if device == "mps" else {}

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
        max_length=512,   # 512 é o limite seguro para 8 GB de memória unificada
        packing=False,
    )

    trainer = SFTTrainer(
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
