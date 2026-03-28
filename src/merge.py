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

# O modelo 9B em bfloat16 completo ocupa ~18 GB.
# Com swap/disco o processo se torna inviável — exige RAM física suficiente.
_MIN_RAM_GB = 16

LORA_HF_DIR  = Path("models/lora-hf")
MERGED_DIR   = Path("models/merged")


def main() -> None:
    if not LORA_HF_DIR.exists():
        raise FileNotFoundError(f"Adaptador LoRA não encontrado em {LORA_HF_DIR}. Execute: make train")

    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    if ram_gb < _MIN_RAM_GB:
        raise MemoryError(
            f"Merge requer ≥ {_MIN_RAM_GB} GB de RAM física. "
            f"Detectado: {ram_gb:.1f} GB.\n"
            f"  → Em Macs com 8 GB, use o adaptador LoRA localmente (make run)\n"
            f"    e publique via uma máquina com mais memória (make docker-push-hf)."
        )
    print(f"  → RAM disponível: {ram_gb:.1f} GB — OK")

    if not settings.hf_token:
        raise ValueError("HF_TOKEN não definido no .env")

    if not settings.hf_push_repo:
        raise ValueError("HF_PUSH_REPO não definido no .env (ex: seu-usuario/nome-do-modelo)")

    print(f"\n→ Carregando modelo base: {settings.model_hf_id}")
    # Merge precisa de pesos em float16/bfloat16 completos — sem quantização 4-bit
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

    print(f"\n→ Fazendo push para o Hugging Face Hub: {settings.hf_push_repo}")
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


if __name__ == "__main__":
    main()
