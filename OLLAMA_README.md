# {ollama_model}

Fine-tuned support assistant based on **Qwen3.5-0.8B**, trained exclusively on help center articles. Answers questions using only the content from its knowledge base — no general knowledge, no hallucination.

---

## Behavior

The model follows three strict rules:

**1. Question clearly covered → answers directly**
```
User: Como configuro a impressora?
Assistant: Para configurar a impressora, acesse Configurações → Dispositivos → Impressoras
e clique em "Adicionar impressora"...
```

**2. Vague or incomplete question → suggests by similarity (brief answer)**
```
User: Impressora não está funcionando, pode ajudar?
Assistant: Pode ser relacionado a configuração de impressora.

Para configurar a impressora, acesse Configurações → Dispositivos...
```

**3. Out-of-scope question → politely refuses**
```
User: Qual é a capital da França?
Assistant: Não tenho essa informação.
```

---

## Usage

```bash
ollama run {ollama_model}
```

### Parameters

| Parameter | Value |
|---|---|
| Context window | 4096 tokens |
| Max tokens (output) | 2048 |
| Temperature | 0.6 |
| Top-p | 0.95 |
| Top-k | 20 |
| Min-p | 0.0 |
| Repeat penalty | 1.0 (disabled) |
| Thinking mode | always on (`<think>` prefill) |

---

## Scope and limitations

- Responds only to questions covered by the training articles
- Does not answer general knowledge questions (geography, cooking, programming, health, etc.)
- Responds in the same language as the user (Portuguese BR by default)
- Knowledge is static — reflects the help center content at training time

---

## Technical details

| Detail | Value |
|---|---|
| Base model | Qwen/Qwen3.5-0.8B |
| Method | QLoRA 4-bit fine-tuning |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| Training epochs | 3 |
| Max sequence length | 2048 tokens |

Built with [lora-smith-qwen](https://github.com/rodrigobrocchi/lora-smith-qwen).
