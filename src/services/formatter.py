from src.config.settings import Settings
from src.domain.models import WPPost
from src.services.html_cleaner import HTMLCleaner

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


class ChatMLFormatter:
    """
    Formata posts do WordPress no formato ChatML do Qwen3.

    O llama-finetune consome texto plano onde os tokens especiais
    já estão injetados diretamente na string (sem JSONL).

    Cada entrada segue o template:
        <|im_start|>system
        {system_prompt}<|im_end|>
        <|im_start|>user
        {titulo_do_artigo}<|im_end|>
        <|im_start|>assistant
        {conteudo_limpo}<|im_end|>
    """

    def __init__(self, settings: Settings) -> None:
        self._system_prompt = settings.system_prompt
        self._cleaner = HTMLCleaner()

    def format_post(self, post: WPPost) -> str | None:
        """
        Converte um WPPost em uma entrada ChatML válida.
        Retorna None se o conteúdo for muito curto para treinar.
        """
        title = self._cleaner.clean(post.title.rendered)
        content = self._cleaner.clean(post.content.rendered)

        if len(content) < 100:
            return None

        entry = (
            f"{_IM_START}system\n"
            f"{self._system_prompt}{_IM_END}\n"
            f"{_IM_START}user\n"
            f"{title}{_IM_END}\n"
            f"{_IM_START}assistant\n"
            f"{content}{_IM_END}\n"
        )
        return entry
