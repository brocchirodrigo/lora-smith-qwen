import random

from src.config.settings import Settings
from src.domain.models import WPPost
from src.services.html_cleaner import HTMLCleaner

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_EOS = "<|endoftext|>"
_THINK_OPEN = "<think>\n"
_THINK_CLOSE = "\n</think>\n\n"


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


# ─── Templates de raciocínio (thinking) ─────────────────────────────────────
# Curtos e variados — o modelo aprende a raciocinar sobre o tema antes de responder.

_THINK_POSITIVE = [
    lambda t: f"O usuário pergunta sobre {t.lower()}. Tenho informações sobre esse tema.",
    lambda t: f"Essa pergunta é sobre {t.lower()}. Encontro conteúdo relevante na base.",
    lambda t: f"Tema identificado: {t.lower()}. Há conteúdo disponível para responder.",
    lambda t: f"A dúvida é sobre {t.lower()}. Consigo responder com o conteúdo da base.",
    lambda t: f"Pergunta sobre {t.lower()}. Tenho material sobre isso.",
]

_THINK_VAGUE = [
    lambda t: f"A pergunta é vaga, mas pode estar relacionada a {t.lower()}. Vou sugerir o tema mais próximo.",
    lambda t: f"Não ficou claro o que o usuário precisa. O tema mais próximo que encontro é {t.lower()}.",
    lambda t: f"Pergunta genérica. Pode ter relação com {t.lower()} na base de conhecimento.",
]


_VARIANTS: list[tuple] = [
    # Variante 1: pergunta direta com título em forma de pergunta
    (
        lambda t: f"{t}?",
        lambda t, c: c,
        "positive",
    ),
    # Variante 2: pedido de ajuda direto
    (
        lambda t: f"Preciso de ajuda com {t.lower()}.",
        lambda t, c: c,
        "positive",
    ),
    # Variante 3: pedido de explicação
    (
        lambda t: f"Me explica sobre {t.lower()}.",
        lambda t, c: c,
        "positive",
    ),
    # Variante 4: relato de problema
    (
        lambda t: f"Não estou conseguindo {t.lower()}, pode me ajudar?",
        lambda t, c: c,
        "positive",
    ),
    # Variante 5: dúvida direta
    (
        lambda t: f"Tenho uma dúvida sobre {t.lower()}.",
        lambda t, c: c,
        "positive",
    ),
    # Variante 6: pergunta genuinamente vaga (usa só a 1ª palavra do título como gatilho)
    (
        lambda t: f"{_first_keyword(t)} não está funcionando, pode ajudar?",
        lambda t, c: f"Pode ser relacionado a {t}.\n\n{_first_paragraph(c)}",
        "vague",
    ),
]


class ChatMLFormatter:
    """
    Formata posts do WordPress no formato ChatML do Qwen3.5 com thinking mode.

    Cada artigo gera 6 entradas com variações naturais do título como
    pergunta do usuário:
      - 5 variantes diretas (v1–v5): resposta é o conteúdo completo
      - 1 variante vaga (v6): pergunta usa só a 1ª palavra do título

    O prompt termina com ``<|im_start|>assistant\n<think>\n`` para alinhar
    com o chat template do Qwen3.5 em ``enable_thinking=true``.

    Cada completion inclui um bloco de raciocínio curto e variado antes da
    resposta, ensinando o modelo a deliberar sobre o tema antes de responder.
    """

    def __init__(self, settings: Settings, seed: int = 42) -> None:
        self._system_prompt = settings.system_prompt
        self._max_content_chars = settings.max_content_chars
        self._cleaner = HTMLCleaner()
        self._rng = random.Random(seed)

    def _truncate(self, text: str) -> str:
        """Trunca no último parágrafo completo dentro do limite de chars."""
        if len(text) <= self._max_content_chars:
            return text
        truncated = text[: self._max_content_chars]
        last_break = truncated.rfind("\n\n")
        if last_break > self._max_content_chars // 2:
            return truncated[:last_break].rstrip()
        return truncated.rstrip()

    def format_post(self, post: WPPost) -> list[dict] | None:
        """
        Converte um WPPost em múltiplas entradas no formato prompt/completion.
        Retorna None se o conteúdo for muito curto para treinar.
        """
        title = self._cleaner.clean(post.title.rendered)
        content = self._truncate(self._cleaner.clean(post.content.rendered))

        if len(content) < 100:
            return None

        system_with_origin = f"{self._system_prompt}\n\n[anota.ai/ajuda: {post.link}]"

        entries = []
        for question_fn, answer_fn, variant_type in _VARIANTS:
            question = question_fn(title)
            answer = answer_fn(title, content)

            if variant_type == "vague":
                think = self._rng.choice(_THINK_VAGUE)(title)
            else:
                think = self._rng.choice(_THINK_POSITIVE)(title)

            entries.append({
                "prompt": (
                    f"{_IM_START}system\n"
                    f"{system_with_origin}{_IM_END}\n"
                    f"{_IM_START}user\n"
                    f"{question}{_IM_END}\n"
                    f"{_IM_START}assistant\n"
                    f"{_THINK_OPEN}"
                ),
                "completion": f"{think}{_THINK_CLOSE}{answer}{_IM_END}\n{_EOS}",
            })
        return entries
