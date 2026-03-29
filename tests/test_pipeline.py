"""
Teste rápido do modelo fine-tuned via pipeline (API de alto nível).

Demonstra o uso mais simples possível do modelo publicado no HF Hub.

Uso:
    uv run python tests/test_pipeline.py
"""

from transformers import pipeline

from src.config.settings import settings

REPO_ID = settings.hf_push_repo
SYSTEM_PROMPT = settings.system_prompt

TEST_QUESTIONS = [
    "Como configurar a impressora?",
    "Cardápio não funciona, pode ajudar?",
    "Qual é a capital da França?",
    "Quero falar com um atendente.",
    "What is machine learning?",
]


def main() -> None:
    print(f"→ Carregando pipeline de {REPO_ID}...\n")
    pipe = pipeline(
        "text-generation",
        model=REPO_ID,
        device_map="auto",
        torch_dtype="bfloat16",
    )

    for i, question in enumerate(TEST_QUESTIONS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        output = pipe(
            messages,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            return_full_text=False,
        )

        response = output[0]["generated_text"]

        # Remove bloco <think>...</think> para exibir apenas a resposta visível
        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()

        print(f"[{i}/{len(TEST_QUESTIONS)}] {question}")
        print(f"  → {response[:300]}")
        print()


if __name__ == "__main__":
    main()
