"""
Merge do adaptador LoRA no modelo base e push para o Hugging Face Hub.

Uso:
    uv run python -m src.merge
    # ou via Makefile:
    make push-hf

Requer no .env:
    HF_TOKEN=hf_...
    HF_PUSH_REPO=seu-usuario/nome-do-modelo
"""

from pathlib import Path

import psutil
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.settings import settings

# O modelo 2B em bfloat16 completo ocupa ~4 GB.
# Verificamos RAM disponível (não total) — outros processos consomem memória.
_MIN_RAM_GB = 6

LORA_HF_DIR  = Path("models/lora-hf")
MERGED_DIR   = Path("models/merged")


def main() -> None:
    if not LORA_HF_DIR.exists():
        raise FileNotFoundError(f"Adaptador LoRA não encontrado em {LORA_HF_DIR}. Execute: make train")

    mem = psutil.virtual_memory()
    ram_available_gb = mem.available / (1024 ** 3)
    ram_total_gb = mem.total / (1024 ** 3)
    if ram_available_gb < _MIN_RAM_GB:
        raise MemoryError(
            f"Merge requer ≥ {_MIN_RAM_GB} GB de RAM disponível. "
            f"Disponível: {ram_available_gb:.1f} GB de {ram_total_gb:.1f} GB.\n"
            f"  → Feche outros processos ou use uma máquina com mais memória livre."
        )
    print(f"  → RAM disponível: {ram_available_gb:.1f} GB de {ram_total_gb:.1f} GB — OK")

    if not settings.hf_token:
        raise ValueError("HF_TOKEN não definido no .env")

    if not settings.hf_push_repo:
        raise ValueError("HF_PUSH_REPO não definido no .env (ex: seu-usuario/nome-do-modelo)")

    print(f"\n→ Carregando modelo base: {settings.model_hf_id}")
    # Merge precisa de pesos em bfloat16 completos — sem quantização 4-bit
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_hf_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.model_hf_id)

    print(f"→ Carregando adaptador LoRA de {LORA_HF_DIR}")
    model = PeftModel.from_pretrained(model, str(LORA_HF_DIR))

    print("→ Fundindo pesos LoRA no modelo base (merge_and_unload)...")
    model = model.merge_and_unload()

    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"→ Salvando modelo fundido em {MERGED_DIR}...")
    model.save_pretrained(str(MERGED_DIR))
    tokenizer.save_pretrained(str(MERGED_DIR))
    print(f"  ✓ Modelo salvo localmente em {MERGED_DIR}")

    print(f"\n→ Fazendo push para o Hugging Face Hub: {settings.hf_push_repo}")
    try:
        model.push_to_hub(
            settings.hf_push_repo,
            token=settings.hf_token,
            private=True,
        )
        tokenizer.push_to_hub(
            settings.hf_push_repo,
            token=settings.hf_token,
            private=True,
        )
        print(f"\n✓ Modelo publicado em: https://huggingface.co/{settings.hf_push_repo}")
    except Exception as exc:
        print(f"\n⚠ Push falhou: {exc}")
        print(f"  O modelo está salvo localmente em {MERGED_DIR.resolve()}")
        print("  Para tentar novamente: make push-hf")
        raise


if __name__ == "__main__":
    main()
