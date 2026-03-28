"""
Gera exemplos negativos (off-topic) para treinar a restrição absoluta do modelo.

Qualquer pergunta fora do conteúdo do WordPress deve receber recusa.
Os exemplos cobrem categorias diversas para que o modelo generalize a rejeição.
"""

import random
from src.config.settings import Settings

_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"

# ─── Banco de perguntas off-topic ─────────────────────────────────────────────
# Categorias: conhecimento geral, programação genérica, matemática, culinária,
# esportes, política, entretenimento, saúde, clima, tradução, conselhos pessoais.

_OFF_TOPIC_PT = [
    # Conhecimento geral
    "Qual é a capital da França?",
    "Quantos planetas existem no sistema solar?",
    "Quem escreveu Dom Casmurro?",
    "Quando foi proclamada a República no Brasil?",
    "Qual é o maior oceano do mundo?",
    "Quem descobriu a penicilina?",
    "O que é fotossíntese?",
    "Qual é a fórmula química da água?",
    "Quantos ossos tem o corpo humano?",
    "Qual é o animal mais rápido do mundo?",
    # Programação genérica
    "Como criar uma lista em Python?",
    "O que é uma API REST?",
    "Como funciona o protocolo HTTP?",
    "O que é recursão em programação?",
    "Como instalar o Node.js no Ubuntu?",
    "Qual é a diferença entre SQL e NoSQL?",
    "O que é um algoritmo de ordenação?",
    "Como funciona o Git?",
    "O que é machine learning?",
    "Como fazer deploy de uma aplicação Docker?",
    # Matemática
    "Quanto é a raiz quadrada de 144?",
    "Resolva: 2x + 5 = 13",
    "Qual é o valor de pi?",
    "Como calcular a área de um triângulo?",
    "O que é um número primo?",
    "Quanto é 15% de 200?",
    "Como converter graus Celsius em Fahrenheit?",
    # Culinária
    "Como fazer bolo de cenoura?",
    "Qual é a receita de brigadeiro?",
    "Como preparar risoto?",
    "Qual é o tempo de cozimento do arroz?",
    "Como fazer pão caseiro?",
    # Esportes
    "Quantas copas do mundo o Brasil ganhou?",
    "Quem é o artilheiro histórico da Champions League?",
    "Quais são as regras do vôlei?",
    "Como funciona o sistema de pontos no tênis?",
    "Quando são os próximos Jogos Olímpicos?",
    # Entretenimento
    "Qual é o filme mais assistido da Netflix?",
    "Quem canta a música 'Evidências'?",
    "Qual é o último episódio de Breaking Bad?",
    "Me recomende um livro de ficção científica.",
    "Quais são as melhores séries do momento?",
    # Saúde e medicina
    "Quais são os sintomas da gripe?",
    "O que é hipertensão arterial?",
    "Como tratar uma dor de cabeça?",
    "Quantas horas de sono um adulto precisa?",
    "O que é diabetes tipo 2?",
    # Clima e geografia
    "Vai chover amanhã em São Paulo?",
    "Qual é a temperatura média do Amazonas?",
    "Quantos estados tem o Brasil?",
    "Qual é o rio mais longo do mundo?",
    # Conselhos pessoais e outros
    "Como me tornar mais produtivo?",
    "Me dê dicas para aprender inglês rápido.",
    "Como economizar dinheiro?",
    "O que fazer quando estou triste?",
    "Como melhorar meu currículo?",
    "Me conte uma piada.",
    "Escreva um poema sobre o outono.",
    "Traduza 'good morning' para o japonês.",
    "Qual é o melhor smartphone do mercado?",
    "Como funciona o Pix?",
]

_OFF_TOPIC_EN = [
    # General knowledge
    "What is the capital of Germany?",
    "How many continents are there?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "What is the speed of light?",
    # Programming
    "How do I reverse a string in JavaScript?",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in Java?",
    "What is a neural network?",
    "How do I center a div in CSS?",
    # Math & science
    "What is the Pythagorean theorem?",
    "How do I calculate compound interest?",
    "What is the periodic table?",
    # Lifestyle
    "What are some tips for better sleep?",
    "How do I lose weight fast?",
    "Give me a recipe for chocolate cake.",
    "What movies are trending right now?",
    "Tell me a fun fact.",
    "Write a haiku about summer.",
    "What is the best programming language to learn?",
]

_ALL_OFF_TOPIC = _OFF_TOPIC_PT + _OFF_TOPIC_EN

# ─── Respostas de recusa variadas ─────────────────────────────────────────────

_REFUSALS_PT = [
    "Não tenho essa informação.",
    "Essa informação não está disponível na minha base de conhecimento.",
    "Não encontrei conteúdo sobre isso nos artigos da central de ajuda.",
    "Não posso responder a isso — meu conhecimento se limita ao conteúdo da central de ajuda.",
    "Essa pergunta está fora do escopo do meu conteúdo de suporte.",
    "Não tenho dados sobre esse assunto.",
    "Isso não está coberto nos materiais que tenho disponíveis.",
]

_REFUSALS_EN = [
    "I don't have that information.",
    "This is not covered in my knowledge base.",
    "I couldn't find anything about this in the help center articles.",
    "I'm unable to answer that — my knowledge is limited to the help center content.",
    "That question is outside the scope of my support content.",
    "I don't have data on that subject.",
    "This topic is not covered in the materials available to me.",
]


def _pick_refusal(question: str) -> str:
    """Escolhe uma resposta de recusa no idioma da pergunta (heurística simples)."""
    q_lower = question.lower()
    use_pt = any(w in q_lower for w in ["qual", "como", "quem", "quando", "quanto", "me ", "uma ", " em ", " do ", " da "])

    pool = _REFUSALS_PT if use_pt else _REFUSALS_EN
    return random.choice(pool)


class NegativeExampleGenerator:
    """
    Gera exemplos de treinamento negativos (off-topic → recusa).

    Cada exemplo segue o mesmo formato ChatML dos exemplos positivos,
    com o system prompt restritivo, uma pergunta fora do escopo e
    uma resposta de recusa variada.
    """

    def __init__(self, settings: Settings, seed: int = 42) -> None:
        self._system_prompt = settings.system_prompt
        self._rng = random.Random(seed)

    def generate(self, n: int) -> list[str]:
        """
        Retorna n exemplos negativos embaralhados.
        Se n > len(banco), repete o banco com variação de resposta.
        """
        questions = self._rng.choices(_ALL_OFF_TOPIC, k=n)
        entries = []
        for q in questions:
            refusal = _pick_refusal(q)
            entry = (
                f"{_IM_START}system\n"
                f"{self._system_prompt}{_IM_END}\n"
                f"{_IM_START}user\n"
                f"{q}{_IM_END}\n"
                f"{_IM_START}assistant\n"
                f"{refusal}{_IM_END}\n"
            )
            entries.append(entry)
        return entries
