# lora-smith-qwen

Pipeline genérico de fine-tuning LoRA para o modelo **Qwen3.5-0.8B** a partir de artigos de qualquer **help center WordPress**.

Treino via Python (PEFT/TRL) em qualquer ambiente — CUDA, Apple Silicon ou CPU. Inferência via **llama.cpp** com GGUF, agnóstico de plataforma.

---

## Execução rápida — ordem das etapas

Para rodar o pipeline completo de uma vez:

```bash
make all
```

Executa em sequência: `clean → setup → download-base → extract → train → export → push`.

Ou etapa por etapa:

```bash
make setup           # 1. instala dependências e compila llama.cpp
make download-base   # 2. baixa o modelo GGUF base (~530 MB)
make extract         # 3. extrai posts do WordPress e gera o dataset
make train           # 4. fine-tuning LoRA (CUDA / MPS / CPU)
make export          # 5. adapter GGUF (llama-cli) + merge local + GGUF fundido (LM Studio)
make run             # 6. chat interativo com o modelo treinado
make push            # 7. (opcional) publica safetensors + GGUF no HF Hub
```

Cada etapa depende da anterior. As etapas 1–3 só precisam ser refeitas se o ambiente ou o conteúdo do WordPress mudar.

---

## Pré-requisitos

### Sistema

| Ferramenta | Versão mínima | Como instalar |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) ou `brew install python` |
| [uv](https://github.com/astral-sh/uv) | qualquer | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| git | qualquer | pré-instalado no macOS/Linux |
| cmake | 3.14+ | `brew install cmake` (macOS) · `apt install cmake` (Debian) |
| build-essential | — | macOS: Xcode CLI · Debian: `apt install build-essential` |
| make | 3.81+ | macOS: `xcode-select --install` · Debian: incluído no `build-essential` |
| Docker | 24+ | [docker.com](https://docker.com) — necessário apenas para o fluxo Docker |

### Hardware mínimo

| Ambiente | RAM mínima | Observações |
|---|---|---|
| Apple Silicon (M1/M2/M3) | 8 GB | Memória unificada. Treino em QLoRA 4-bit via MPS |
| Linux + CUDA | 6 GB VRAM | QLoRA 4-bit via bitsandbytes |
| Linux CPU | 24 GB RAM | Treino em fp32, lento mas funcional |
| Docker | 6 GB VRAM (GPU) | Requer Linux + CUDA. MPS não é suportado em container |

> **`make export` (merge):** requer ≥ 4 GB de RAM **disponível** — o modelo 0.8B em bf16 completo ocupa ~2 GB.

### Espaço em disco

| Item | Tamanho |
|---|---|
| Modelo GGUF base (inferência) | ~1.5 GB |
| Modelo HF (treino, em cache) | ~4 GB |
| Adaptador LoRA HF gerado | ~50 MB |
| Adaptador LoRA GGUF (llama-cli) | ~50 MB |
| Modelo fundido safetensors (merge) | ~4 GB |
| Modelo fundido GGUF Q4_K_M (LM Studio) | ~1.5 GB |

---

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Gerenciador de pacotes | `uv` |
| HTTP / API WordPress | `httpx` (async) |
| Validação | `pydantic` + `pydantic-settings` |
| Limpeza HTML | `beautifulsoup4` |
| Download / publicação de modelos | `huggingface-hub` |
| Treino LoRA | `transformers` + `peft` + `trl` |
| Quantização 4-bit | `bitsandbytes` |
| Motor de inferência | `llama-cli` (llama.cpp, compilado localmente) |

---

## Estrutura do projeto

```
lora-smith-qwen/
├── .env                       # Variáveis de ambiente (não commitado)
├── .env.example               # Template documentado
├── Dockerfile                 # Imagem Docker para treinamento em CUDA
├── docker-compose.yml         # Compose com volumes para modelos e dados
├── Makefile                   # Orquestrador do pipeline (local e Docker)
├── pyproject.toml
├── HF_README.md               # README publicado no Hugging Face Hub (make push-hf)
├── prompts/
│   └── prompts.yaml           # System prompt com persona cooperativa
├── data/
│   └── processed/
│       ├── train.jsonl        # Dataset de treino (ChatML em JSONL)
│       └── valid.jsonl        # Dataset de validação
├── models/
│   ├── base/                  # Modelo GGUF base (inferência via llama-cli)
│   ├── lora-hf/               # Adaptador LoRA no formato HuggingFace
│   ├── lora/                  # Adaptador LoRA GGUF (inferência via llama-cli)
│   ├── merged/                # Modelo fundido safetensors (HF Hub / LM Studio source)
│   └── merged-q4km.gguf       # Modelo fundido quantizado (LM Studio)
├── scripts/
│   └── llama.cpp/             # Repositório compilado no setup
└── src/
    ├── config/settings.py     # Pydantic Settings (lê .env)
    ├── domain/models.py       # WPPost dataclass
    ├── infrastructure/
    │   └── wp_client.py       # Cliente WordPress REST API
    ├── services/
    │   ├── html_cleaner.py
    │   ├── formatter.py       # ChatMLFormatter — 6 variantes por artigo
    │   └── negative_generator.py  # Exemplos negativos (~206 perguntas off-topic e adjacentes)
    ├── extract.py             # Extração, formatação e mix de exemplos negativos
    ├── train.py               # Fine-tuning LoRA (CUDA/MPS/CPU)
    └── merge.py               # Merge LoRA + base (local e push para HF Hub)
```

---

## Configuração

Copie e edite o arquivo de variáveis:
```bash
cp .env.example .env
```

| Variável | Default | Descrição |
|---|---|---|
| `WP_BASE_URL` | `https://example.com/wp-json/wp/v2` | URL da API WordPress REST do seu site |
| `WP_USERNAME` | *(vazio)* | Usuário (opcional, API pública) |
| `WP_APP_PASSWORD` | *(vazio)* | Senha de aplicativo do WordPress |
| `MAX_POSTS` | `0` (todos) | Limite de posts a extrair |
| `MODEL_HF_ID` | `Qwen/Qwen3.5-0.8B` | Modelo HuggingFace para treino |
| `MODEL_REPO_ID` | `unsloth/Qwen3.5-0.8B-GGUF` | Repositório do GGUF base |
| `MODEL_FILENAME` | `Qwen3.5-0.8B-Q4_K_M.gguf` | Arquivo GGUF local |
| `TRAIN_ITERS` | `1000` | Hard cap de steps (0 = sem limite, usa só `TRAIN_EPOCHS`) |
| `TRAIN_EPOCHS` | `3` | Épocas alvo — steps calculados automaticamente pelo tamanho do dataset |
| `MAX_CONTENT_CHARS` | `2500` | Limite de chars do conteúdo por entrada (~714 tokens, garante que o token de fim nunca seja truncado) |
| `CPU_THREADS` | `6` | Threads para compilação e inferência |
| `HF_TOKEN` | *(vazio)* | Token HF com permissão de escrita — necessário para `make push` |
| `HF_PUSH_REPO` | *(vazio)* | Repositório de destino no HF Hub (ex: `seu-usuario/nome-do-modelo`) |

O system prompt e a persona do modelo ficam em `prompts/prompts.yaml` — edite diretamente sem tocar no `.env`.

---

## Execução local (macOS / Linux)

### 1. Setup

```bash
make setup
```

- Cria o ambiente virtual Python com `uv`
- Instala todas as dependências Python
- Clona e compila o `llama-cli` em `scripts/llama.cpp/`

### 2. Download do modelo base

```bash
make download-base
```

- Baixa `Qwen3.5-0.8B-Q4_K_M.gguf` (~530 MB) para `models/base/`
- Usado exclusivamente para inferência com `llama-cli`

### 3. Extração dos dados

```bash
make extract
```

- Consulta a API REST do WordPress e pagina todos os posts publicados
- Formata cada artigo em **6 variantes** no formato `prompt/completion` para label masking nativo:
  - 5 variantes diretas: "Título?", "Preciso de ajuda com...", "Me explica sobre...", "Não estou conseguindo...", "Tenho uma dúvida sobre..." — resposta direta sem prefixo
  - 1 variante vaga: usa só a 1ª palavra do título como gatilho → resposta com "Pode ser relacionado a [título]." + 1º parágrafo, treinando sugestão por similaridade apenas quando a pergunta é imprecisa
- Cada entrada positiva inclui a **URL de origem do artigo** (`post.link`) no bloco system como âncora de treinamento (`[anota.ai/ajuda: URL]`). Como o loss é calculado apenas na completion, a URL não afeta a saída — ela ancora o modelo no domínio real da Anota AI durante o treino. Exemplos negativos não possuem essa âncora, reforçando o contraste entre escopo e fora-de-escopo.
- Gera automaticamente exemplos **negativos** (off-topic e adjacentes → recusa) em ~206 perguntas de 15 categorias, correspondendo a 60% do total de positivos (mínimo 40)
- Embaralha positivos e negativos antes de dividir
- Salva em `data/processed/train.jsonl` (90%) e `valid.jsonl` (10%)

> Para sites privados, configure `WP_USERNAME` e `WP_APP_PASSWORD` no `.env`.

### 4. Treinamento LoRA

```bash
make train
```

- Baixa `Qwen/Qwen3.5-0.8B` do HuggingFace (~4 GB, fica em cache após o primeiro uso)
- Aplica QLoRA 4-bit via `bitsandbytes` (CUDA) ou bfloat16 sem quantização (MPS/CPU)
- Calcula automaticamente o número de steps com base em `TRAIN_EPOCHS` e o tamanho do dataset — `TRAIN_ITERS` funciona como hard cap
- LoRA aplicado em todas as camadas lineares (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) com rank 64 e alpha 128
- Sequências de até 1280 tokens; conteúdo truncado a 2500 chars antes de formatar para garantir que o token de fim nunca seja cortado
- Salva `generation_config.json` junto ao adaptador em `models/lora-hf/` com `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.05`, `repetition_penalty=1.0`, `max_new_tokens=2048` — lido automaticamente pelo Transformers e pelo LM Studio

| Ambiente | Modo | Memória usada |
|---|---|---|
| CUDA | QLoRA 4-bit | ~4–6 GB VRAM |
| Apple Silicon (MPS) | bfloat16 | ~5 GB memória unificada |
| CPU | float32 | ~16 GB RAM |

### 5. Export

```bash
make export
```

Executa três etapas em sequência:

1. **`export-lora`** — converte `models/lora-hf/` para `models/lora/adapter.gguf` (usado pelo `make run`)
2. **`merge`** — funde o adaptador LoRA no modelo base e salva em `models/merged/` (safetensors, bfloat16)
3. **`export-merged-gguf`** — converte `models/merged/` para `models/merged-q4km.gguf` (GGUF Q4_K_M, usado pelo LM Studio)

> Requer ≥ 6 GB de RAM disponível para o passo de merge.

### 6. Inferência interativa

```bash
make run
```

- Chat com `llama-cli` usando o modelo GGUF base + adaptador LoRA GGUF
- System prompt carregado automaticamente de `prompts/prompts.yaml`
- O modelo abre o chat com uma saudação e aguarda a pergunta

### 7. Publicação (opcional)

```bash
make push
```

Publica em sequência:

1. **HF Hub safetensors** — requer `HF_TOKEN` + `HF_PUSH_REPO`
2. **HF Hub GGUF** — mesmo repositório, arquivo `merged-q4km.gguf`

```
seu-usuario/nome-do-modelo/   ← HF Hub
├── model.safetensors
├── tokenizer.json
├── config.json
└── merged-q4km.gguf
```

Para publicar individualmente:
```bash
make push-hf        # apenas safetensors no HF (requer make merge)
make push-gguf      # apenas GGUF no HF (requer make export)
```

---

## Execução via Docker (Linux + CUDA)

O Docker é indicado para ambientes Linux com GPU Nvidia. **No Mac, o container não acessa MPS/Metal** — o treino cai para CPU dentro do container.

Os modelos e dados ficam na sua máquina e são montados como volumes — a imagem Docker contém apenas o ambiente compilado.

### 1. Preparação local (uma vez)

```bash
make setup           # compila llama.cpp local (para inferência no Mac)
make download-base   # baixa o GGUF base em models/base/
```

### 2. Build da imagem

```bash
make docker-build
```

- Constrói a imagem `lora-smith-qwen` com CUDA 12.1, deps Python e llama.cpp compilado

### 3. Pipeline no container

```bash
make docker-extract   # extrai posts → data/processed/
make docker-train     # treina LoRA → models/lora-hf/
make docker-export    # adapter GGUF + merge + GGUF fundido
make docker-push      # publica safetensors + GGUF no HF Hub
make docker-run       # chat interativo com GPU
```

---

## Limpeza

```bash
make clean
```

Remove tudo: dataset, adapters, modelos fundidos, GGUF base e cache HuggingFace. Após o `make clean`, o próximo ciclo começa do `make download-base`.

---

## Persona e escopo do modelo

O modelo é treinado com uma **persona de suporte** que segue três regras de comportamento:

1. **Pergunta coberta → responde diretamente** com o conteúdo disponível.
2. **Pergunta vaga ou incompleta → sugere por similaridade**: identifica o tema mais próximo e inicia com "Pode ser relacionado a [tema]." antes de responder.
3. **Pergunta completamente fora do escopo → recusa** com "Não tenho essa informação."

Isso é garantido por três mecanismos combinados:

1. **System prompt genérico** (`prompts/prompts.yaml`): instrui o modelo a responder apenas com base no conteúdo disponível na sua base de conhecimento, sem conhecimento externo.
2. **Âncora de origem por artigo**: cada entrada positiva inclui a URL real do artigo (`[site/ajuda: URL]`) no bloco system — o modelo aprende a associar seu conhecimento a conteúdos concretos do help center. Exemplos negativos não têm âncora, reforçando o contraste de escopo.
3. **Exemplos negativos no dataset**: ~206 perguntas em 15 categorias (off-topic geral + adjacentes de suporte), cada uma pareada com uma resposta de recusa variada no idioma correto.

O Qwen3.5 usa **thinking mode nativo** na inferência — o modelo raciocina em um bloco `<think>` antes de responder. Isso é ativado via `--chat-template-kwargs '{"enable_thinking":true}'` no `make run` e via chat template no LM Studio.

Para ajustar o comportamento, edite `prompts/prompts.yaml` (persona e regras de escopo), `src/services/formatter.py` (variantes de pergunta e âncora de origem) e `src/services/negative_generator.py` (banco de perguntas e frases de recusa).

---

## Observações técnicas

- **Por que GGUF para inferência?** O `llama-cli` roda em qualquer plataforma (macOS, Linux, Windows, Docker) sem dependências de Python ou CUDA. O `adapter.gguf` é carregado em runtime junto ao modelo base, sem fundir os pesos.
- **Por que o treino não usa llama-finetune?** O Qwen3.5 é uma arquitetura híbrida SSM+Transformer (Gated Delta Net). O `llama-finetune` não suporta backward pass para camadas SSM — o treino via Python+PEFT contorna essa limitação.
- **Por que todas as camadas lineares?** Conhecimento factual é codificado nas camadas MLP (`gate_proj`, `up_proj`, `down_proj`). Treinar apenas atenção (`q_proj`, `v_proj`) ensina formato e comportamento, mas não memoriza conteúdo — o modelo segue as regras do system prompt mas não sabe responder perguntas específicas da base de conhecimento.
- **Merge vs. adaptador:** `make export` gera dois artefatos — o `adapter.gguf` (~50 MB) para uso local com `make run`, e o `merged-q4km.gguf` (~1.5 GB) para uso standalone no LM Studio ou distribuição. O modelo fundido não precisa do adaptador para rodar.
- **Cache HuggingFace:** o modelo de treino (~4 GB) fica em `~/.cache/huggingface/`. Para liberar disco: `huggingface-cli delete-cache`.
- **Parâmetros de inferência:** `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.05`, `repeat_penalty=1.0`. Definidos em dois pontos independentes: `generation_config.json` (Transformers/LM Studio safetensors) e flags do `llama-cli`/`make run`. Runtimes que ignoram o `generation_config.json` (ex.: LM Studio carregando GGUF diretamente) precisam configurar manualmente.
