"""
Teste completo do modelo fine-tuned via Transformers (carrega do HF Hub).

Executa perguntas positivas, vagas, negativas, adjacentes e multi-idioma,
exibindo a resposta visível (sem bloco <think>) e um resumo final.

Uso:
    uv run python tests/test_model.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.settings import settings

REPO_ID = settings.hf_push_repo
SYSTEM_PROMPT = settings.system_prompt

TEST_CASES = [
    # ── Positivas — devem ser respondidas com conteúdo da base ──
    ("positiva", "Como configurar a impressora na Anota AI?"),
    ("positiva", "Preciso de ajuda com primeiros passos com a Anota AI."),
    ("positiva", "Como funciona o atendimento pelo WhatsApp?"),
    ("positiva", "Como cadastrar um produto no cardápio?"),
    ("positiva", "Me explica sobre pagamento online."),
    # ── Vagas — devem sugerir tema relacionado ──
    ("vaga", "Impressora não está funcionando, pode ajudar?"),
    ("vaga", "Cardápio não está funcionando, pode ajudar?"),
    ("vaga", "Pedido deu problema, o que faço?"),
    # ── Negativas off-topic — devem recusar ──
    ("negativa", "Qual é a capital da França?"),
    ("negativa", "Como fazer bolo de cenoura?"),
    ("negativa", "Me explica o que é machine learning."),
    ("negativa", "Quanto é 15% de 200?"),
    ("negativa", "Who wrote Romeo and Juliet?"),
    # ── Adjacentes (parecem suporte mas fora do escopo) — devem recusar ──
    ("adjacente", "Como cancelo minha assinatura?"),
    ("adjacente", "Quero falar com um atendente."),
    ("adjacente", "Quanto custa o plano empresarial?"),
    ("adjacente", "O sistema ficou fora do ar ontem, o que aconteceu?"),
    ("adjacente", "Vocês integram com a Shopify?"),
    # ── Multi-idioma — devem recusar em inglês ──
    ("negativa_en", "What is the best programming language to learn?"),
    ("negativa_en", "How do I center a div in CSS?"),
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

    stop_ids = [
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        tokenizer.convert_tokens_to_ids("<|im_start|>"),
    ]
    eos_token_id = [tid for tid in stop_ids if tid is not None]
    print(f"  Stop token IDs: {eos_token_id}")
    print(f"  Total de testes: {len(TEST_CASES)}\n")

    results = []

    for i, (category, question) in enumerate(TEST_CASES, 1):
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

        full_response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        # Separa thinking da resposta visível
        thinking = ""
        visible = full_response
        if "</think>" in full_response:
            parts = full_response.split("</think>", 1)
            thinking = parts[0].replace("<think>", "").strip()
            visible = parts[1].strip()

        results.append({
            "category": category,
            "question": question,
            "thinking": thinking,
            "visible": visible,
        })

        print(f"{'='*70}")
        print(f"  [{i}/{len(TEST_CASES)}] ({category}) {question}")
        if thinking:
            print(f"  THINK: {thinking[:120]}")
        print(f"{'─'*70}")
        print(f"  {visible[:500]}")
        print()

    # ── Resumo ──
    print(f"\n{'='*70}")
    print(f"  RESUMO")
    print(f"{'='*70}")
    for cat in ["positiva", "vaga", "negativa", "adjacente", "negativa_en"]:
        cat_results = [r for r in results if r["category"] == cat]
        if not cat_results:
            continue
        print(f"\n  [{cat.upper()}] ({len(cat_results)} testes)")
        for r in cat_results:
            short = r["visible"][:80].replace("\n", " ")
            print(f"    Q: {r['question'][:50]}")
            print(f"    R: {short}...")


if __name__ == "__main__":
    main()
