from src.config.settings import Settings
from src.domain.models import WPPost
from src.services.html_cleaner import HTMLCleaner

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_EOS = "<|endoftext|>"

_VARIANTS: list[tuple] = [
    (
        lambda t: t,
        lambda t, c: c,
    ),
    (
        lambda t: f"Preciso de ajuda com {t.lower()}.",
        lambda t, c: c,
    ),
    (
        lambda t: f"Me explica sobre {t.lower()}.",
        lambda t, c: c,
    ),
    (
        lambda t: f"Não estou conseguindo {t.lower()}, pode me ajudar?",
        lambda t, c: f"Pode ser relacionado a {t}.\n\n{c}",
    ),
    (
        lambda t: f"Tenho uma dúvida sobre {t.lower()}.",
        lambda t, c: f"Pode ser relacionado a {t}.\n\n{c}",
    ),
]


class ChatMLFormatter:
    """
    Formata posts do WordPress no formato ChatML do Qwen3.

    Cada artigo gera 5 entradas com variações naturais do título como
    pergunta do usuário:
      - 3 variantes diretas: resposta é o conteúdo puro
      - 2 variantes vagas: resposta é prefixada com "Pode ser relacionado a [título]."
        para treinar a persona cooperativa de sugestão por similaridade

    Formato de cada entrada:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {variante_da_pergunta}<|im_end|>
        <|im_start|>assistant
        {resposta}<|im_end|>
        <|endoftext|>

    O <|endoftext|> final ensina o modelo que a sequência termina completamente
    após o turno do assistant. Sem ele, o modelo não aprende a parar.

    O conteúdo é truncado em max_content_chars para garantir que o <|im_end|>
    final nunca seja cortado pelo max_length do trainer.
    """

    def __init__(self, settings: Settings) -> None:
        self._system_prompt = settings.system_prompt
        self._max_content_chars = settings.max_content_chars
        self._cleaner = HTMLCleaner()

    def _truncate(self, text: str) -> str:
        """Trunca no último parágrafo completo dentro do limite de chars."""
        if len(text) <= self._max_content_chars:
            return text
        truncated = text[: self._max_content_chars]
        last_break = truncated.rfind("\n\n")
        if last_break > self._max_content_chars // 2:
            return truncated[:last_break].rstrip()
        return truncated.rstrip()

    def format_post(self, post: WPPost) -> list[str] | None:
        """
        Converte um WPPost em múltiplas entradas ChatML.
        Retorna None se o conteúdo for muito curto para treinar.
        """
        title = self._cleaner.clean(post.title.rendered)
        content = self._truncate(self._cleaner.clean(post.content.rendered))

        if len(content) < 100:
            return None

        entries = []
        for question_fn, answer_fn in _VARIANTS:
            question = question_fn(title)
            answer = answer_fn(title, content)
            entry = (
                f"{_IM_START}system\n"
                f"{self._system_prompt}{_IM_END}\n"
                f"{_IM_START}user\n"
                f"{question}{_IM_END}\n"
                f"{_IM_START}assistant\n"
                f"{answer}{_IM_END}\n"
                f"{_EOS}"
            )
            entries.append(entry)
        return entries
