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

_MIN_RAM_GB = 6

LORA_HF_DIR  = Path("models/lora-hf")
MERGED_DIR   = Path("models/merged")


def merge_local() -> None:
    """Funde o adaptador LoRA no modelo base e salva em models/merged (sem push)."""
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

    print(f"\n→ Carregando modelo base: {settings.model_hf_id}")
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_hf_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cpu",
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


def push_to_hub() -> None:
    """Publica models/merged no Hugging Face Hub (sem re-mergear)."""
    if not (MERGED_DIR / "config.json").exists():
        raise FileNotFoundError(f"Modelo fundido não encontrado em {MERGED_DIR}. Execute: make merge")

    if not settings.hf_token:
        raise ValueError("HF_TOKEN não definido no .env")

    if not settings.hf_push_repo:
        raise ValueError("HF_PUSH_REPO não definido no .env (ex: seu-usuario/nome-do-modelo)")

    from transformers import AutoModelForCausalLM as _M, AutoTokenizer as _T
    from huggingface_hub import HfApi
    model = _M.from_pretrained(str(MERGED_DIR), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = _T.from_pretrained(str(MERGED_DIR))

    api = HfApi()
    print(f"\n→ Limpando repositório existente: {settings.hf_push_repo}")
    try:
        api.delete_repo(repo_id=settings.hf_push_repo, token=settings.hf_token, repo_type="model")
    except Exception:
        pass
    api.create_repo(repo_id=settings.hf_push_repo, token=settings.hf_token, private=False, exist_ok=True)

    print(f"→ Publicando no Hugging Face Hub: {settings.hf_push_repo}")
    try:
        model.push_to_hub(settings.hf_push_repo, token=settings.hf_token, private=False)
        tokenizer.push_to_hub(settings.hf_push_repo, token=settings.hf_token, private=True)

        readme_content = Path("HF_README.md").read_text(encoding="utf-8").format(
            repo_id=settings.hf_push_repo,
            base_model=settings.model_hf_id,
        )
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=settings.hf_push_repo,
            token=settings.hf_token,
        )
        print(f"\n✓ Modelo publicado em: https://huggingface.co/{settings.hf_push_repo}")
    except Exception as exc:
        print(f"\n⚠ Push falhou: {exc}")
        raise


def main() -> None:
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "merge"
    if cmd == "push":
        push_to_hub()
    else:
        merge_local()


if __name__ == "__main__":
    main()
