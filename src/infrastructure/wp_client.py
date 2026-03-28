import base64
from typing import AsyncGenerator

import httpx
from rich.console import Console

from src.config.settings import Settings
from src.domain.models import WPPost

console = Console()


class WordPressClient:
    """Cliente HTTP assíncrono para a API REST do WordPress."""

    def __init__(self, settings: Settings) -> None:
        self._base_url = settings.wp_base_url.rstrip("/")
        self._max_posts = settings.max_posts
        self._headers = self._build_headers(settings)

    def _build_headers(self, settings: Settings) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        if settings.wp_username and settings.wp_app_password:
            credentials = f"{settings.wp_username}:{settings.wp_app_password}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"
        return headers

    async def iter_posts(self) -> AsyncGenerator[WPPost, None]:
        """Itera sobre todos os posts publicados via paginação automática."""
        page = 1
        per_page = 100
        total_fetched = 0

        async with httpx.AsyncClient(headers=self._headers, timeout=30.0) as client:
            while True:
                params: dict[str, str | int] = {
                    "status": "publish",
                    "per_page": per_page,
                    "page": page,
                    "_fields": "id,title,content,status,link",
                }

                response = await client.get(
                    f"{self._base_url}/posts", params=params
                )

                if response.status_code == 400:
                    break

                response.raise_for_status()
                posts_data = response.json()

                if not posts_data:
                    break

                for post_data in posts_data:
                    post = WPPost(**post_data)
                    yield post
                    total_fetched += 1

                    if self._max_posts > 0 and total_fetched >= self._max_posts:
                        console.print(
                            f"[yellow]Limite de {self._max_posts} posts atingido.[/yellow]"
                        )
                        return

                total_header = response.headers.get("X-WP-Total", "?")
                console.print(
                    f"[dim]Página {page} processada. "
                    f"Total acumulado: {total_fetched}/{total_header}[/dim]"
                )

                if len(posts_data) < per_page:
                    break

                page += 1
