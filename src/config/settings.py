from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROMPTS_FILE = Path(__file__).parents[2] / "prompts" / "prompts.yaml"


def _load_prompts() -> dict:
    if PROMPTS_FILE.exists():
        return yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8")) or {}
    return {}


class Settings(BaseSettings):
    """Configuração centralizada lida do arquivo .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    wp_base_url: str = ""
    wp_username: str = ""
    wp_app_password: str = ""
    max_posts: int = 0

    model_hf_id: str = "Qwen/Qwen3.5-2B"
    model_filename: str = "Qwen3.5-2B-Q4_K_M.gguf"

    train_iters: int = 1000
    cpu_threads: int = 6

    hf_token: str = ""
    hf_push_repo: str = ""

    @property
    def system_prompt(self) -> str:
        return _load_prompts().get("system", "").strip()


settings = Settings()
