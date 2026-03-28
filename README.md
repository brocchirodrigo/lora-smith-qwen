# lora-smith-qwen

Pipeline genérico de fine-tuning LoRA para o modelo **Qwen3.5-2B** a partir de artigos de qualquer **help center WordPress**.

Treino via Python (PEFT/TRL) em qualquer ambiente — CUDA, Apple Silicon ou CPU. Inferência via **llama.cpp** com GGUF, agnóstico de plataforma.

---

## Execução rápida — ordem das etapas

```bash
make setup           # 1. instala dependências e compila llama.cpp
make download-base   # 2. baixa o modelo GGUF base (~5.4 GB)
make extract         # 3. extrai posts do WordPress e gera o dataset
make train           # 4. fine-tuning LoRA (CUDA / MPS / CPU)
make export-lora     # 5. converte adaptador HF → GGUF
make run             # 6. chat interativo com o modelo treinado
make push-hf         # 7. (opcional) merge + publicação no HF Hub — exige ≥ 16 GB RAM
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
| make | 3.81+ | macOS: `xcode-select --install` (incluído no Xcode CLI) · Debian: incluído no `build-essential` |
| Docker | 24+ | [docker.com](https://docker.com) — necessário apenas para o fluxo Docker |

### Hardware mínimo

| Ambiente | RAM mínima | Observações |
|---|---|---|
| Apple Silicon (M1/M2/M3) | 8 GB | Memória unificada. Treino em QLoRA 4-bit via MPS |
| Linux + CUDA | 4 GB VRAM | QLoRA 4-bit via bitsandbytes |
| Linux CPU | 24 GB RAM | Treino em fp32, lento mas funcional |
| Docker | 8 GB VRAM (GPU) | Requer Linux + CUDA. MPS não é suportado em container |

> **`make push-hf` (merge):** requer ≥ 6 GB de RAM — o modelo 2B em bf16 completo ocupa ~4 GB.

### Espaço em disco

| Item | Tamanho |
|---|---|
| Modelo GGUF base (inferência) | ~1.5 GB |
| Modelo HF (treino, em cache) | ~4 GB |
| Adaptador LoRA HF gerado | ~50 MB |
| Adaptador LoRA GGUF gerado | ~15 MB |
| Modelo fundido (merge, bf16) | ~4 GB |

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
| Saída de terminal | `rich` |

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
├── prompts/
│   └── prompts.yaml           # System prompt restritivo
├── data/
│   └── processed/
│       ├── train.jsonl        # Dataset de treino (ChatML em JSONL)
│       └── valid.jsonl        # Dataset de validação
├── models/
│   ├── base/                  # Modelo GGUF base (inferência)
│   ├── lora-hf/               # Adaptador LoRA no formato HuggingFace
│   ├── lora/                  # Adaptador LoRA no formato GGUF (inferência)
│   └── merged/                # Modelo fundido (base + LoRA), pronto para o HF Hub
├── scripts/
│   └── llama.cpp/             # Repositório compilado no setup
└── src/
    ├── config/settings.py     # Pydantic Settings (lê .env)
    ├── domain/models.py       # WPPost dataclass
    ├── infrastructure/
    │   └── wp_client.py       # Cliente WordPress REST API
    ├── services/
    │   ├── html_cleaner.py
    │   ├── formatter.py       # ChatMLFormatter — exemplos positivos (WP → resposta)
    │   └── negative_generator.py  # Exemplos negativos (off-topic → recusa)
    ├── extract.py             # Extração, formatação e mix de exemplos negativos
    ├── train.py               # Fine-tuning LoRA (CUDA/MPS/CPU)
    └── merge.py               # Merge LoRA + base e push para o Hugging Face Hub
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
| `MODEL_HF_ID` | `Qwen/Qwen3.5-2B` | Modelo HuggingFace para treino |
| `MODEL_REPO_ID` | `unsloth/Qwen3.5-2B-GGUF` | Repositório do GGUF base |
| `MODEL_FILENAME` | `Qwen3.5-2B-Q4_K_M.gguf` | Arquivo GGUF local |
| `TRAIN_ITERS` | `500` | Iterações de treino |
| `CPU_THREADS` | `6` | Threads para compilação e inferência |
| `HF_TOKEN` | *(vazio)* | Token HF com permissão de escrita — necessário para `make push-hf` |
| `HF_PUSH_REPO` | *(vazio)* | Repositório de destino no HF Hub (ex: `seu-usuario/nome-do-modelo`) |

O system prompt fica em `prompts/prompts.yaml` — edite diretamente sem tocar no `.env`.

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

- Baixa `Qwen3.5-2B-Q4_K_M.gguf` (~1.5 GB) para `models/base/`
- Usado exclusivamente para inferência com `llama-cli`

### 3. Extração dos dados

```bash
make extract
```

- Consulta a API REST do WordPress e pagina todos os posts publicados
- Formata cada artigo em ChatML (system / user / assistant) — **exemplos positivos**
- Gera automaticamente exemplos **negativos** (off-topic → recusa) correspondendo a 25% do total de positivos (mínimo 30), cobrindo perguntas de conhecimento geral, programação, matemática, culinária, esportes, saúde e outros domínios
- Embaralha positivos e negativos antes de dividir
- Salva em `data/processed/train.jsonl` (95%) e `valid.jsonl` (5%)

> Para sites privados, configure `WP_USERNAME` e `WP_APP_PASSWORD` no `.env`.

### 4. Treinamento LoRA

```bash
make train
```

- Baixa `Qwen/Qwen3.5-2B` do HuggingFace (~4 GB, fica em cache após o primeiro uso)
- Aplica QLoRA 4-bit via `bitsandbytes` (~2 GB em memória)
- Treina por 500 iterações com `SFTTrainer`
- Salva o adaptador em `models/lora-hf/`

| Ambiente | Modo | Memória usada |
|---|---|---|
| CUDA | QLoRA 4-bit | ~2–3 GB VRAM |
| Apple Silicon (MPS) | QLoRA 4-bit | ~2.5 GB memória unificada |
| CPU | float32 | ~8 GB RAM |

### 5. Exportação LoRA → GGUF

```bash
make export-lora
```

- Converte `models/lora-hf/` para `models/lora/adapter.gguf` (~50 MB)
- Usa `convert_lora_to_gguf.py` do llama.cpp

### 6. Publicação no Hugging Face Hub (opcional)

```bash
make push-hf
```

Requer `HF_TOKEN` e `HF_PUSH_REPO` no `.env`.

- Carrega o modelo base em **bfloat16 completo** (sem quantização) — necessário para fundir os pesos
- Funde o adaptador LoRA no modelo base com `merge_and_unload()` — o resultado é um modelo independente, sem dependência do adaptador
- Salva o modelo fundido em `models/merged/`
- Faz push para o HF Hub como repositório **privado**

> **Hardware:** o merge requer ~6 GB de RAM livre (o modelo 2B em bf16 completo ocupa ~4 GB).

### 7. Inferência interativa

```bash
make run
```

- Chat com `llama-cli` usando o modelo GGUF base + adaptador LoRA GGUF
- System prompt carregado automaticamente de `prompts/prompts.yaml`

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
make docker-extract       # extrai posts → data/processed/
make docker-train         # treina LoRA → models/lora-hf/
make docker-export-lora   # converte → models/lora/adapter.gguf
make docker-push-hf       # merge + push para o HF Hub (requer HF_TOKEN e HF_PUSH_REPO)
make docker-run           # chat interativo com GPU
```

---

## Limpeza

```bash
make clean
```

Remove `data/processed/`, `models/lora/`, `models/lora-hf/` e `models/merged/`. Mantém o modelo GGUF base e o cache HuggingFace intactos.

---

## Restrição absoluta de escopo

O modelo é treinado para responder **exclusivamente** com base no conteúdo extraído do WordPress. Qualquer pergunta fora desse escopo recebe uma recusa explícita.

Isso é garantido por dois mecanismos combinados:

1. **System prompt restritivo** (`prompts/prompts.yaml`): instrui o modelo a nunca usar conhecimento externo e a listar categorias proibidas explicitamente (programação genérica, matemática, culinária, saúde etc.)

2. **Exemplos negativos no dataset**: durante o `make extract`, são gerados automaticamente exemplos de perguntas off-topic pareadas com respostas de recusa variadas (em pt-BR e en). O modelo aprende a recusar — não apenas a responder no estilo correto.

Para ajustar o comportamento de recusa, edite `prompts/prompts.yaml` (system prompt) e `src/services/negative_generator.py` (banco de perguntas e frases de recusa).

---

## Observações técnicas

- **Por que GGUF para inferência?** O `llama-cli` roda em qualquer plataforma (macOS, Linux, Windows, Docker) sem dependências de Python ou CUDA. O `adapter.gguf` é carregado em runtime junto ao modelo base, sem fundir os pesos.
- **Por que o treino não usa llama-finetune?** O Qwen3.5 é uma arquitetura híbrida SSM+Transformer (Gated Delta Net). O `llama-finetune` não suporta backward pass para camadas SSM — o treino via Python+PEFT contorna essa limitação.
- **Merge vs. adaptador:** `make push-hf` funde os pesos LoRA no modelo base — o resultado não precisa do adaptador para rodar. `make run` usa o adaptador separado via llama.cpp, o que é mais leve localmente.
- **Cache HuggingFace:** o modelo de treino (~18 GB) fica em `~/.cache/huggingface/`. Para liberar disco: `huggingface-cli delete-cache`.
- **Tamanho do adaptador:** o LoRA treina uma fração pequena dos 2B parâmetros. O `adapter.gguf` resultante tem ~15 MB.
