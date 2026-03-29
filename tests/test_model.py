"""
Teste do modelo fine-tuned via Transformers (carrega do HF Hub).

Uso:
    uv run python scripts/test_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.settings import settings

REPO_ID = settings.hf_push_repo
SYSTEM_PROMPT = settings.system_prompt

TEST_QUESTIONS = [
    # Positivas — devem ser respondidas com conteudo da base
    "Como configurar a impressora na Anota AI?",
    "Preciso de ajuda com primeiros passos com a Anota AI.",
    "Como funciona o atendimento pelo WhatsApp?",
    # Vagas — devem sugerir tema relacionado
    "Impressora não está funcionando, pode ajudar?",
    "Cardápio não está funcionando, pode ajudar?",
    # Negativas — devem ser recusadas
    "Qual é a capital da França?",
    "Como fazer bolo de cenoura?",
    "Me explica o que é machine learning.",
    "Como cancelo minha assinatura?",
    "Quero falar com um atendente.",
]


def main() -> None:
    print(f"→ Carregando modelo de {REPO_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        REPO_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)
    print(f"  Dispositivo: {model.device}")

    # Stop tokens: <|im_end|>, <|endoftext|>, <|im_start|>
    stop_ids = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
    ]
    eos_token_id = [tid for tid in stop_ids if tid is not None]
    print(f"  Stop token IDs: {eos_token_id}\n")

    for i, question in enumerate(TEST_QUESTIONS, 1):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0.05,
                repetition_penalty=1.0,
                eos_token_id=eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # Remove bloco <think>...</think> para exibir apenas a resposta visivel
        if "</think>" in response:
            response = response.split("</think>", 1)[1].strip()

        print(f"{'='*60}")
        print(f"  [{i}/{len(TEST_QUESTIONS)}] {question}")
        print(f"{'─'*60}")
        print(f"  {response[:500]}")
        print()


if __name__ == "__main__":
    main()
