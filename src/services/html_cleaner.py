import re

from bs4 import BeautifulSoup

_NOISE_PATTERNS = re.compile(
    r"(leia\s+tamb[eé]m[\s:]+.*|artigos?\s+relacionados?[\s:]+.*"
    r"|compartilhe\s+(este|esse)\s+artigo.*|veja\s+tamb[eé]m[\s:]+.*"
    r"|[úu]ltima\s+atualiza[çc][ãa]o[\s:]+.*)",
    re.IGNORECASE | re.MULTILINE,
)


class HTMLCleaner:
    """Limpa e normaliza HTML de artigos WordPress."""

    def clean(self, html: str) -> str:
        """Converte HTML bruto em texto plain limpo e normalizado."""
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "iframe", "figure", "noscript",
                         "nav", "header", "footer", "aside", "form"]):
            tag.decompose()
        for tag in soup.find_all(class_=re.compile(
            r"(breadcrumb|related|share|social|comment|sidebar|widget|menu|nav|"
            r"tag|categor|author|date|meta|pagination|search)", re.IGNORECASE
        )):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = _NOISE_PATTERNS.sub("", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text
