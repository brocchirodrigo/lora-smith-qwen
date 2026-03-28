# lora-smith-qwen

Pipeline genérico de fine-tuning LoRA para o modelo **Qwen3.5-9B** a partir de artigos de qualquer **help center WordPress**.

Treino via Python (PEFT/TRL) em qualquer ambiente — CUDA, Apple Silicon ou CPU. Inferência via **llama.cpp** com GGUF, agnóstico de plataforma.

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
| Apple Silicon (M1/M2/M3) | 18 GB | Memória unificada. Treino em QLoRA 4-bit via MPS |
| Linux + CUDA | 8 GB VRAM | QLoRA 4-bit via bitsandbytes |
| Linux CPU | 24 GB RAM | Treino em fp32, lento mas funcional |
| Docker | 8 GB VRAM (GPU) | Requer Linux + CUDA. MPS não é suportado em container |

### Espaço em disco

| Item | Tamanho |
|---|---|
| Modelo GGUF base (inferência) | ~5.4 GB |
| Modelo HF (treino, em cache) | ~18 GB |
| Adaptador LoRA HF gerado | ~200 MB |
| Adaptador LoRA GGUF gerado | ~50 MB |

---

## Stack

| Camada | Tecnologia |
|---|---|
| Linguagem | Python 3.11+ |
| Gerenciador de pacotes | `uv` |
| HTTP / API WordPress | `httpx` (async) |
| Validação | `pydantic` + `pydantic-settings` |
| Limpeza HTML | `beautifulsoup4` |
| Download de modelos | `huggingface-hub` |
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
│   └── prompts.yaml           # System prompt e variações
├── data/
│   └── processed/
│       ├── train.jsonl        # Dataset de treino (ChatML em JSONL)
│       └── valid.jsonl        # Dataset de validação
├── models/
│   ├── base/                  # Modelo GGUF base (inferência)
│   ├── lora-hf/               # Adaptador LoRA no formato HuggingFace
│   └── lora/                  # Adaptador LoRA no formato GGUF (inferência)
├── scripts/
│   └── llama.cpp/             # Repositório compilado no setup
└── src/
    ├── config/settings.py     # Pydantic Settings (lê .env)
    ├── domain/models.py       # WPPost dataclass
    ├── infrastructure/
    │   └── wp_client.py       # Cliente WordPress REST API
    ├── services/
    │   ├── html_cleaner.py
    │   └── formatter.py       # ChatMLFormatter
    ├── extract.py             # Extração e formatação do dataset
    └── train.py               # Fine-tuning LoRA (CUDA/MPS/CPU)
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
| `MODEL_HF_ID` | `Qwen/Qwen3.5-9B` | Modelo HuggingFace para treino |
| `MODEL_REPO_ID` | `unsloth/Qwen3.5-9B-GGUF` | Repositório do GGUF base |
| `MODEL_FILENAME` | `Qwen3.5-9B-Q4_K_M.gguf` | Arquivo GGUF local |
| `TRAIN_ITERS` | `500` | Iterações de treino |
| `CPU_THREADS` | `6` | Threads para compilação e inferência |

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

- Baixa `Qwen3.5-9B-Q4_K_M.gguf` (~5.4 GB) para `models/base/`
- Usado exclusivamente para inferência com `llama-cli`

### 3. Extração dos dados

```bash
make extract
```

- Consulta a API REST do WordPress e pagina todos os posts publicados
- Formata cada artigo em ChatML (system / user / assistant)
- Salva em `data/processed/train.jsonl` (95%) e `valid.jsonl` (5%)

> Para sites privados, configure `WP_USERNAME` e `WP_APP_PASSWORD` no `.env`.

### 4. Treinamento LoRA

```bash
make train
```

- Baixa `Qwen/Qwen3.5-9B` do HuggingFace (~18 GB, fica em cache após o primeiro uso)
- Aplica QLoRA 4-bit via `bitsandbytes` (~4.5 GB em memória)
- Treina por 500 iterações com `SFTTrainer`
- Salva o adaptador em `models/lora-hf/`

| Ambiente | Modo | Memória usada |
|---|---|---|
| CUDA | QLoRA 4-bit | ~6–8 GB VRAM (cap em 6.5 GB) |
| Apple Silicon (MPS) | QLoRA 4-bit | ~6 GB memória unificada |
| CPU | float32 | ~36 GB RAM |

### 5. Exportação LoRA → GGUF

```bash
make export-lora
```

- Converte `models/lora-hf/` para `models/lora/adapter.gguf` (~50 MB)
- Usa `convert_lora_to_gguf.py` do llama.cpp

### 6. Inferência interativa

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
make docker-run           # chat interativo com GPU
```

---

## Limpeza

```bash
make clean
```

Remove `data/processed/`, `models/lora/` e `models/lora-hf/`. Mantém o modelo GGUF base e o cache HuggingFace intactos.

---

## Observações técnicas

- **Por que GGUF para inferência?** O `llama-cli` roda em qualquer plataforma (macOS, Linux, Windows, Docker) sem dependências de Python ou CUDA. O `adapter.gguf` é carregado em runtime junto ao modelo base, sem fundir os pesos.
- **Por que o treino não usa llama-finetune?** O Qwen3.5 é uma arquitetura híbrida SSM+Transformer (Gated Delta Net). O `llama-finetune` não suporta backward pass para camadas SSM — o treino via Python+PEFT contorna essa limitação.
- **Cache HuggingFace:** o modelo de treino (~18 GB) fica em `~/.cache/huggingface/`. Para liberar disco: `huggingface-cli delete-cache`.
- **Tamanho do adaptador:** o LoRA treina apenas 43M de 9B parâmetros (0.48%). O `adapter.gguf` resultante tem ~50 MB.
