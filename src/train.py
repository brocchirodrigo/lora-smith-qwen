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
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import SFTConfig, SFTTrainer

from src.config.settings import settings

TRAIN_FILE   = Path("data/processed/train.jsonl")
VALID_FILE   = Path("data/processed/valid.jsonl")
LORA_HF_DIR  = Path("models/lora-hf")


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
        load_kwargs.update({
            "torch_dtype": torch.bfloat16,
            "device_map": {"": "mps"},
        })
        print("  → Modo: bfloat16 (Apple Silicon MPS, sem quantização 4-bit)")

    else:
        torch.set_num_threads(max(1, settings.cpu_threads - 2))
        os.environ["OMP_NUM_THREADS"] = str(max(1, settings.cpu_threads - 2))
        load_kwargs.update({"torch_dtype": torch.float32, "low_cpu_mem_usage": True})
        print(f"  → Modo: float32 (CPU, threads: {max(1, settings.cpu_threads - 2)})")

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

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
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    print("→ Carregando datasets...")
    train_dataset = load_jsonl(TRAIN_FILE)
    valid_dataset = load_jsonl(VALID_FILE) if VALID_FILE.exists() else None
    print(f"  Treino: {len(train_dataset)} | Validação: {len(valid_dataset) if valid_dataset else 0}")

    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        use_bf16 = major >= 8
    elif device == "mps":
        use_bf16 = True
    else:
        use_bf16 = False
    grad_accum = 8
    gc_kwargs = {"use_reentrant": False} if device in ("cuda", "mps") else {}

    effective_batch = grad_accum
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    target_steps = settings.train_epochs * steps_per_epoch
    max_steps = min(target_steps, settings.train_iters) if settings.train_iters > 0 else target_steps
    warmup_steps = max(10, max_steps // 10)
    eval_steps = max(10, max_steps // 10)
    save_steps = eval_steps
    print(f"  Dataset: {len(train_dataset)} | Steps/época: {steps_per_epoch} | "
          f"Épocas alvo: {settings.train_epochs} | Total steps: {max_steps} | "
          f"Label masking: ativo (loss só no assistant)")

    training_args = SFTConfig(
        output_dir=str(LORA_HF_DIR),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs=gc_kwargs,
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        dataloader_pin_memory=(device == "cuda"),
        dataloader_num_workers=0,
        eval_strategy="steps" if valid_dataset else "no",
        eval_steps=eval_steps if valid_dataset else None,
        completion_only_loss=True,
        max_length=2048,
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

    print(f"\n→ Iniciando fine-tuning ({max_steps} steps)...")
    trainer.train()

    print(f"\n→ Salvando adaptador LoRA em {LORA_HF_DIR}...")
    trainer.save_model(str(LORA_HF_DIR))
    tokenizer.save_pretrained(str(LORA_HF_DIR))

    # generation_config.json — parâmetros recomendados para tarefas factuais/precisas.
    # Lido pelo Transformers e pelo LM Studio (safetensors).
    GenerationConfig(
        temperature=0.6,
        do_sample=True,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        repetition_penalty=1.0,
        max_new_tokens=2048,
    ).save_pretrained(str(LORA_HF_DIR))

    print(f"\n✓ Adaptador HF salvo em: {LORA_HF_DIR}")
    print("  Próximo passo: make export-lora")


if __name__ == "__main__":
    main()
