from src.config.settings import Settings
from src.domain.models import WPPost
from src.services.html_cleaner import HTMLCleaner

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_EOS = "<|endoftext|>"


def _first_paragraph(text: str) -> str:
    """Retorna o primeiro parágrafo não vazio com pelo menos 50 chars."""
    for para in text.split("\n\n"):
        para = para.strip()
        if len(para) >= 50:
            return para
    return text


_STRIP_CHARS = ".,;:?!"

_SKIP_WORDS = {"como", "o", "a", "os", "as", "qual", "quais", "quando", "onde",
               "por", "para", "que", "um", "uma", "de", "do", "da", "em",
               "meu", "minha", "meus", "minhas", "seu", "sua", "seus", "suas",
               "fazer", "ver", "usar", "ter", "ser", "está", "não"}


def _first_keyword(title: str) -> str:
    """Retorna a primeira palavra significativa do título (ignora artigos e pronomes interrogativos)."""
    for word in title.split():
        w = word.rstrip(_STRIP_CHARS).lower()
        if w not in _SKIP_WORDS and len(w) >= 3:
            return word.rstrip(_STRIP_CHARS).capitalize()
    # fallback: primeira palavra sem filtro
    word = title.split()[0] if title.split() else title
    return word.rstrip(_STRIP_CHARS).capitalize()


_VARIANTS: list[tuple] = [
    # Variante 1: pergunta direta com título em forma de pergunta
    # (era: título exato → generalização fraca; agora usa "?" para simular pergunta natural)
    (
        lambda t: f"{t}?",
        lambda t, c: c,
    ),
    # Variante 2: pedido de ajuda direto
    (
        lambda t: f"Preciso de ajuda com {t.lower()}.",
        lambda t, c: c,
    ),
    # Variante 3: pedido de explicação
    (
        lambda t: f"Me explica sobre {t.lower()}.",
        lambda t, c: c,
    ),
    # Variante 4: relato de problema — pergunta específica, resposta direta sem prefixo
    # (era: "Pode ser relacionado a..." — mas a pergunta não é vaga, então o prefixo era incorreto)
    (
        lambda t: f"Não estou conseguindo {t.lower()}, pode me ajudar?",
        lambda t, c: c,
    ),
    # Variante 5: dúvida direta — sem prefixo
    # (era: "Pode ser relacionado a..." — mesma razão acima)
    (
        lambda t: f"Tenho uma dúvida sobre {t.lower()}.",
        lambda t, c: c,
    ),
    # Variante 6: pergunta genuinamente vaga (usa só a 1ª palavra do título como gatilho)
    # → ensina o modelo a usar "Pode ser relacionado a" apenas quando a pergunta é imprecisa
    # → resposta usa só o 1º parágrafo: adiciona variação de profundidade (resposta curta)
    (
        lambda t: f"{_first_keyword(t)} não está funcionando, pode ajudar?",
        lambda t, c: f"Pode ser relacionado a {t}.\n\n{_first_paragraph(c)}",
    ),
]


class ChatMLFormatter:
    """
    Formata posts do WordPress no formato ChatML do Qwen3.

    Cada artigo gera 6 entradas com variações naturais do título como
    pergunta do usuário:
      - 5 variantes diretas (v1–v5): resposta é o conteúdo completo, sem prefixo
      - 1 variante vaga (v6): pergunta usa só a 1ª palavra do título (genuinamente
        imprecisa), resposta usa "Pode ser relacionado a [título]." + 1º parágrafo
        — treina tanto o padrão de sugestão quanto respostas curtas

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
