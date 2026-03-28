import re

from bs4 import BeautifulSoup


class HTMLCleaner:
    """Limpa e normaliza HTML de artigos WordPress."""

    _STRIP_PATTERN = re.compile(r"(^\s+)|(\s+$)")

    def clean(self, html: str) -> str:
        """Converte HTML bruto em texto plain limpo e normalizado."""
        if not html:
            return ""

        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "iframe", "figure", "noscript"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()

        return text
