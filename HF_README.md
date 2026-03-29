---
library_name: transformers
tags:
- support
- chat
- peft
- qlora
- fine-tuned
- llama-cpp
license: mit
language:
- pt
base_model:
- {base_model}
pipeline_tag: text-generation
---

# Assistente de Suporte (Qwen3.5 Fine-tuned)

Modelo fine-tuned a partir de **{base_model}** via QLoRA para atuar como assistente de suporte especializado, com conhecimento restrito à base de artigos do help center.

O modelo responde perguntas usando exclusivamente o conteúdo dos artigos extraídos durante o treinamento. Perguntas fora do escopo são recusadas de forma educada.

## Uso

### LM Studio

Baixe o arquivo `merged-q4km.gguf` diretamente pela interface do LM Studio pesquisando por `{repo_id}`.

> **Thinking mode:** o template padrão do Qwen3.5 desativa thinking por padrão. No LM Studio, acesse as configurações do modelo e habilite **"Thinking"** (ou equivalente) manualmente para garantir que os blocos `<think>` sejam gerados.

### llama-cli (llama.cpp)

```bash
llama-cli \
  --model merged-q4km.gguf \
  --conversation \
  -c 4096 \
  --temp 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.00 \
  --repeat-penalty 1.0 \
  --chat-template-kwargs '{{"enable_thinking":true}}' \
  -sys "Você é um assistente de suporte com conhecimento restrito à base de conhecimento disponível. Nunca invente informações."
```

### Transformers (Python)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    "{repo_id}",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

messages = [
    {{"role": "system", "content": "Você é um assistente de suporte..."}},
    {{"role": "user", "content": "Como faço para configurar X?"}},
]
# enable_thinking=True: o template padrão do Qwen3.5 desativa thinking por padrão em modelos pequenos.
# add_generation_prompt=True prefills <think> no turno do assistant.
# temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.0 vêm do generation_config.json.
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True))
```

## Arquivos disponíveis

| Arquivo | Formato | Uso |
|---|---|---|
| `model.safetensors` | safetensors (bfloat16) | Fine-tuning adicional via Python |
| `merged-q4km.gguf` | GGUF Q4_K_M | LM Studio · llama-cli · inferência local |

## Detalhes do treinamento

| Parâmetro | Valor |
|---|---|
| Modelo base | `{base_model}` |
| Método | QLoRA 4-bit (bitsandbytes) |
| LoRA rank | 16 |
| LoRA alpha | 16 |
| LoRA dropout | 0.1 |
| Módulos treinados | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Comprimento máximo | 2048 tokens |
| Épocas | 3 (steps calculados pelo tamanho do dataset) |
| Scheduler | Cosine |

## Limitações

- O modelo responde apenas perguntas cobertas pelo conteúdo dos artigos de treinamento.
- Não deve ser usado para tarefas gerais (escrita criativa, código, matemática, saúde, etc.).
- As respostas são limitadas ao idioma e escopo do conteúdo original.

## Pipeline

Gerado com [lora-smith-qwen](https://github.com/rodrigobrocchi/lora-smith-qwen) — pipeline de fine-tuning LoRA para help centers WordPress.
