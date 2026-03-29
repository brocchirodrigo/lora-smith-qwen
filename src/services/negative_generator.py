"""
Gera exemplos negativos (off-topic) para treinar a restrição absoluta do modelo.

Qualquer pergunta fora do conteúdo disponível deve receber recusa.
Os exemplos cobrem categorias diversas para que o modelo generalize a rejeição.
"""

import random
from src.config.settings import Settings

_IM_START = "<|im_start|>"
_IM_END   = "<|im_end|>"
_EOS      = "<|endoftext|>"

# ─── Banco de perguntas off-topic ─────────────────────────────────────────────

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
    "Qual é a montanha mais alta do mundo?",
    "Quem foi Albert Einstein?",
    "O que é a teoria da relatividade?",
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
    "O que é um banco de dados relacional?",
    "Como funciona o React?",
    "O que é uma variável em programação?",
    # Matemática
    "Quanto é a raiz quadrada de 144?",
    "Resolva: 2x + 5 = 13",
    "Qual é o valor de pi?",
    "Como calcular a área de um triângulo?",
    "O que é um número primo?",
    "Quanto é 15% de 200?",
    "Como converter graus Celsius em Fahrenheit?",
    "O que é uma derivada?",
    "Como calcular juros compostos?",
    # Culinária
    "Como fazer bolo de cenoura?",
    "Qual é a receita de brigadeiro?",
    "Como preparar risoto?",
    "Qual é o tempo de cozimento do arroz?",
    "Como fazer pão caseiro?",
    "Qual a diferença entre fritar e refogar?",
    "Como temperar uma frigideira de ferro?",
    # Esportes
    "Quantas copas do mundo o Brasil ganhou?",
    "Quem é o artilheiro histórico da Champions League?",
    "Quais são as regras do vôlei?",
    "Como funciona o sistema de pontos no tênis?",
    "Quando são os próximos Jogos Olímpicos?",
    "Qual é o recorde mundial dos 100 metros rasos?",
    "Como funciona o VAR no futebol?",
    # Entretenimento
    "Qual é o filme mais assistido da Netflix?",
    "Quem canta a música 'Evidências'?",
    "Qual é o último episódio de Breaking Bad?",
    "Me recomende um livro de ficção científica.",
    "Quais são as melhores séries do momento?",
    "Qual foi o vencedor do Oscar de melhor filme?",
    "Me indique um podcast de tecnologia.",
    # Saúde e medicina
    "Quais são os sintomas da gripe?",
    "O que é hipertensão arterial?",
    "Como tratar uma dor de cabeça?",
    "Quantas horas de sono um adulto precisa?",
    "O que é diabetes tipo 2?",
    "Qual é a dose diária recomendada de vitamina C?",
    "Quais são os benefícios da meditação?",
    "Como controlar a ansiedade?",
    # Clima e geografia
    "Vai chover amanhã em São Paulo?",
    "Qual é a temperatura média do Amazonas?",
    "Quantos estados tem o Brasil?",
    "Qual é o rio mais longo do mundo?",
    "Qual é o país mais populoso do mundo?",
    "Qual é o fuso horário de Brasília?",
    # Finanças pessoais
    "Como funciona o Tesouro Direto?",
    "O que é CDB?",
    "Como declarar imposto de renda?",
    "Qual é a melhor corretora de investimentos?",
    "O que é FII?",
    "Como funciona o FGTS?",
    "Como sair das dívidas?",
    "Qual é a diferença entre ações e fundos de investimento?",
    # História
    "Quando terminou a Segunda Guerra Mundial?",
    "Quem foi Napoleão Bonaparte?",
    "O que foi a Revolução Industrial?",
    "Quando o homem pisou na Lua?",
    "O que foi o Iluminismo?",
    "Quem foram os Maias?",
    "O que foi a Guerra Fria?",
    # Viagens
    "Quais são os documentos necessários para viajar para a Europa?",
    "Preciso de visto para ir ao Japão?",
    "Qual é a melhor época para visitar Machu Picchu?",
    "Quais são as principais atrações de Lisboa?",
    "Como é o metrô de Londres?",
    "Me recomende hotéis baratos em Buenos Aires.",
    # Animais e pets
    "Como adestrar um cachorro?",
    "Qual é a expectativa de vida de um gato?",
    "O que um labrador pode comer?",
    "Como tratar uma infecção em peixe de aquário?",
    "Quais vacinas um cachorro precisa tomar por ano?",
    # Juridico e burocracia
    "Como registrar uma empresa no Brasil?",
    "O que é habeas corpus?",
    "Como funciona a CLT?",
    "O que é usufruto?",
    "Como funciona o seguro-desemprego?",
    "O que é CNPJ?",
    # Ciência e tecnologia
    "O que é inteligência artificial?",
    "Como funciona o GPS?",
    "O que é energia solar?",
    "Como funciona a internet?",
    "O que é blockchain?",
    "Como funciona um motor elétrico?",
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
    "Me dê uma dica de leitura para este mês.",
    "Como aprender a tocar violão?",
]

_OFF_TOPIC_EN = [
    # General knowledge
    "What is the capital of Germany?",
    "How many continents are there?",
    "Who wrote Romeo and Juliet?",
    "What year did World War II end?",
    "What is the speed of light?",
    "Who was Isaac Newton?",
    "What is the Big Bang theory?",
    # Programming
    "How do I reverse a string in JavaScript?",
    "What is the difference between TCP and UDP?",
    "How does garbage collection work in Java?",
    "What is a neural network?",
    "How do I center a div in CSS?",
    "What is the difference between Git merge and rebase?",
    # Math & science
    "What is the Pythagorean theorem?",
    "How do I calculate compound interest?",
    "What is the periodic table?",
    "What is quantum physics?",
    # Finance
    "What is an index fund?",
    "How does the stock market work?",
    "What is cryptocurrency?",
    "How do I file taxes in the US?",
    # Travel & lifestyle
    "What are the best places to visit in Italy?",
    "Do I need a visa to visit Canada?",
    "What are some tips for better sleep?",
    "How do I lose weight fast?",
    "Give me a recipe for chocolate cake.",
    "What movies are trending right now?",
    "Tell me a fun fact.",
    "Write a haiku about summer.",
    "What is the best programming language to learn?",
    "How do I train my dog to sit?",
    "What is the life expectancy of a parrot?",
    # History
    "Who was Julius Caesar?",
    "What was the French Revolution?",
    "When did the Berlin Wall fall?",
]


# ─── Perguntas adjacentes — soam como suporte mas estão fora do escopo ────────
# Esse banco é crítico para evitar scope leak: cobre requests legítimos de suporte
# que um assistente de base de conhecimento não deve responder (atendimento humano,
# pricing, incidentes, feature requests, reclamações formais).

_ADJACENT_PT = [
    # Atendimento humano
    "Quero falar com um atendente.",
    "Tem como falar com alguém da equipe de suporte?",
    "Preciso de atendimento humano urgente.",
    "Como entro em contato com o suporte por telefone?",
    "Tem chat ao vivo disponível?",
    # Feature requests
    "Podem adicionar a opção de exportar para PDF?",
    "Quando vai ter integração com o Google Drive?",
    "Vocês planejam lançar modo escuro?",
    "Seria possível adicionar notificações por e-mail?",
    "Quando vai sair a próxima atualização do sistema?",
    # Incidentes e status
    "O sistema ficou fora do ar ontem à noite, o que aconteceu?",
    "Quando vai voltar a funcionar normalmente?",
    "O app está com lentidão hoje?",
    "Teve alguma manutenção programada recentemente?",
    # Billing / assinatura
    "Como cancelo minha assinatura?",
    "Quanto custa o plano empresarial?",
    "Tem desconto para estudantes ou ONGs?",
    "Como emito nota fiscal da minha assinatura?",
    "Posso fazer upgrade do meu plano agora?",
    "Qual é a política de reembolso?",
    # Dados / incidente grave
    "Perdi todos os meus dados, o que aconteceu?",
    "O sistema deletou meu arquivo sem aviso.",
    "Minha conta foi acessada por outra pessoa.",
    # Feedback e reclamação
    "Quero registrar uma reclamação formal.",
    "Quero fazer uma sugestão de melhoria.",
    "Tenho um elogio para a equipe de vocês.",
    "Como envio um feedback oficial?",
    # Comparação / concorrência
    "Qual é a diferença de vocês para o concorrente X?",
    "Vocês têm parceria com outras ferramentas do mercado?",
    "Por que devo escolher vocês em vez de outro sistema?",
    # Hardware e Ecossistema externo
    "Como instalo a impressora Bematech b805?",
    "Como configuro a impressora padrão no Windows?",
    "Como faço integração com a Nuvemshop?",
    "Vocês integram com a Shopify?",
    "Como configuro meu cardápio no iFood?",
    "Qual maquininha de cartão vocês recomendam, Stone ou PagSeguro?",
    "Como funciona o sistema da Totvs?",
    "Como emitir cupom fiscal na Bematech?",
    "Minha impressora térmica parou de funcionar, o que eu faço?",
]

_ADJACENT_EN = [
    # Human support
    "I want to talk to a human agent.",
    "How do I contact support by phone?",
    "Is there live chat available?",
    # Feature requests
    "Can you add PDF export functionality?",
    "When will Google Drive integration be available?",
    "Are you planning to release a dark mode?",
    "When is the next system update coming out?",
    # Incidents
    "The system was down last night, what happened?",
    "When will it be back to normal?",
    "Is the app running slow today?",
    # Billing
    "How do I cancel my subscription?",
    "What is the enterprise plan pricing?",
    "Is there a student discount?",
    "What is your refund policy?",
    # Data / serious incident
    "I lost all my data, what happened?",
    "The app deleted my file without warning.",
    # Feedback
    "I want to file a formal complaint.",
    "I have a suggestion for improvement.",
    "How do I send official feedback?",
    # Comparison
    "How are you different from competitor X?",
    "Why should I choose you over other systems?",
]


_ALL_OFF_TOPIC = _OFF_TOPIC_PT + _OFF_TOPIC_EN + _ADJACENT_PT + _ADJACENT_EN

# ─── Respostas de recusa variadas ─────────────────────────────────────────────

_REFUSALS_PT = [
    "Não tenho essa informação.",
    "Essa informação não está disponível na minha base de conhecimento.",
    "Não encontrei conteúdo sobre isso na minha base de conhecimento.",
    "Não posso responder a isso — meu conhecimento se limita ao conteúdo disponível.",
    "Essa pergunta está fora do escopo do conteúdo disponível.",
    "Não tenho dados sobre esse assunto.",
    "Isso não está coberto nos materiais que tenho disponíveis.",
]

_REFUSALS_EN = [
    "I don't have that information.",
    "This is not covered in my knowledge base.",
    "I couldn't find anything about this in my knowledge base.",
    "I'm unable to answer that — my knowledge is limited to the available content.",
    "That question is outside the scope of the available content.",
    "I don't have data on that subject.",
    "This topic is not covered in the materials available to me.",
]


# Primeiras palavras típicas de perguntas em inglês.
# Detecção por first-word evita falsos positivos de substring (ex: "tem" em "system").
_EN_STARTERS = {
    "how", "what", "when", "where", "who", "why", "is", "are", "can",
    "do", "does", "will", "would", "i", "the", "this", "that", "it",
    # imperative starters
    "give", "tell", "write", "show", "find", "explain", "translate",
    "help", "describe", "list", "make", "create", "get",
}


def _pick_refusal(question: str, rng: random.Random | None = None) -> str:
    """Escolhe uma resposta de recusa no idioma da pergunta.

    Usa a primeira palavra como sinal de idioma: perguntas em inglês quase
    sempre começam com uma das palavras de _EN_STARTERS; as demais são PT.
    """
    first_word = question.lower().split()[0].rstrip(".,;:?!") if question.split() else ""
    pool = _REFUSALS_EN if first_word in _EN_STARTERS else _REFUSALS_PT
    return (rng or random).choice(pool)


class NegativeExampleGenerator:
    """
    Gera exemplos de treinamento negativos (off-topic → recusa).

    Retorna dicts {"prompt": ..., "completion": ...} alinhados com o formato
    do ChatMLFormatter para que o SFTTrainer aplique label masking consistente.
    """

    def __init__(self, settings: Settings, seed: int = 42) -> None:
        self._system_prompt = settings.system_prompt
        self._rng = random.Random(seed)

    def generate(self, n: int) -> list[dict]:
        """
        Retorna n exemplos negativos embaralhados.
        Se n > len(banco), repete o banco com variação de resposta.
        """
        questions = self._rng.choices(_ALL_OFF_TOPIC, k=n)
        entries = []
        for q in questions:
            refusal = _pick_refusal(q, rng=self._rng)
            entries.append({
                "prompt": (
                    f"{_IM_START}system\n"
                    f"{self._system_prompt}{_IM_END}\n"
                    f"{_IM_START}user\n"
                    f"{q}{_IM_END}\n"
                    f"{_IM_START}assistant\n"
                ),
                "completion": f"{refusal}{_IM_END}\n{_EOS}",
            })
        return entries
