# ─── Detecção de SO ──────────────────────────────────────────────────────────
OS_NAME := $(shell uname -s)

ifeq ($(OS_NAME), Darwin)
  GGML_METAL := ON
  GGML_CUDA  := OFF
else
  GGML_METAL := OFF
  CUDA_AVAILABLE := $(shell command -v nvcc 2>/dev/null && echo ON || echo OFF)
  GGML_CUDA  := $(CUDA_AVAILABLE)
endif

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR     := data/processed
TRAIN_FILE   := $(DATA_DIR)/train.jsonl
MODEL_BASE   := models/base
MODEL_LORA_HF := models/lora-hf
MODEL_LORA   := models/lora/adapter.gguf
LLAMA_DIR    := scripts/llama.cpp
LLAMA_BIN    := $(LLAMA_DIR)/build/bin

-include .env
export

CPU_THREADS   ?= 6
TRAIN_ITERS   ?= 500
MODEL_HF_ID   ?= Qwen/Qwen3.5-9B
MODEL_REPO_ID ?= unsloth/Qwen3.5-9B-GGUF
MODEL_FILENAME ?= Qwen3.5-9B-Q4_K_M.gguf

MODEL_MERGED := models/merged
HF_PUSH_REPO ?=
HF_TOKEN     ?=

MODEL_GGUF := $(MODEL_BASE)/$(MODEL_FILENAME)

.PHONY: help setup download-base extract train export-lora push-hf run clean \
        docker-build docker-extract docker-train docker-export-lora docker-push-hf docker-run

# ─── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  lora-smith-qwen — Pipeline de Fine-Tuning LoRA (GGUF / ambiente agnóstico)"
	@echo "  SO: $(OS_NAME)  |  Metal: $(GGML_METAL)  |  CUDA: $(GGML_CUDA)"
	@echo ""
	@echo "  Comandos disponíveis (local):"
	@echo "    make setup              Instala dependências Python e compila llama.cpp"
	@echo "    make download-base      Baixa o modelo GGUF base do Hugging Face"
	@echo "    make extract            Extrai posts do WordPress e gera JSONL"
	@echo "    make train              Fine-tuning LoRA via Python (CUDA/MPS/CPU)"
	@echo "    make export-lora        Converte adaptador HF → GGUF (llama.cpp)"
	@echo "    make push-hf            Merge LoRA + base e publica no Hugging Face Hub"
	@echo "    make run                Chat interativo com llama-cli + LoRA GGUF"
	@echo "    make clean              Remove dados processados e adaptadores"
	@echo ""
	@echo "  Comandos Docker (Linux + CUDA):"
	@echo "    make docker-build       Constrói a imagem Docker"
	@echo "    make docker-extract     Extrai posts dentro do container"
	@echo "    make docker-train       Fine-tuning dentro do container (GPU)"
	@echo "    make docker-export-lora Converte adaptador HF → GGUF no container"
	@echo "    make docker-push-hf     Merge LoRA + push para HF Hub no container"
	@echo "    make docker-run         Chat interativo no container"
	@echo ""

# ─── Setup ────────────────────────────────────────────────────────────────────
setup:
	@echo "→ Criando estrutura de diretórios..."
	@mkdir -p $(DATA_DIR) $(MODEL_BASE) models/lora $(MODEL_LORA_HF) scripts tmp

	@echo "→ Criando ambiente virtual e instalando dependências Python com uv..."
	@uv venv --quiet --seed 2>/dev/null || true
	@uv pip install -e . --quiet

	@echo "→ Clonando llama.cpp (se necessário)..."
	@if [ ! -d "$(LLAMA_DIR)" ]; then \
		git clone --depth=1 https://github.com/ggerganov/llama.cpp $(LLAMA_DIR); \
	else \
		echo "  llama.cpp já existe, pulando clone."; \
	fi

	@echo "→ Compilando llama-cli..."
	@if ! command -v cmake > /dev/null 2>&1; then \
		if [ "$(OS_NAME)" = "Darwin" ]; then \
			echo "✗ cmake não encontrado. Instale com: brew install cmake"; exit 1; \
		else \
			sudo apt-get install -y cmake build-essential; \
		fi; \
	fi
	@mkdir -p $(LLAMA_DIR)/build
	@cd $(LLAMA_DIR)/build && \
		cmake .. -DGGML_METAL=$(GGML_METAL) -DGGML_CUDA=$(GGML_CUDA) -DLLAMA_CURL=OFF && \
		cmake --build . --config Release --target llama-cli -j$(CPU_THREADS)

	@echo ""
	@echo "✓ Setup concluído! Próximo passo: make download-base"

# ─── Download do modelo base ──────────────────────────────────────────────────
download-base:
	@echo "→ Baixando $(MODEL_FILENAME) de $(MODEL_REPO_ID)..."
	@uv run python -c "\
from huggingface_hub import hf_hub_download; \
import os; \
repo = os.environ.get('MODEL_REPO_ID', 'unsloth/Qwen3.5-9B-GGUF'); \
fname = '$(MODEL_FILENAME)'; \
hf_hub_download(repo, fname, local_dir='$(MODEL_BASE)'); \
print('✓ Modelo GGUF salvo em: $(MODEL_GGUF)')"

# ─── Extração de dados ────────────────────────────────────────────────────────
extract:
	@echo "→ Extraindo artigos do WordPress..."
	@uv run python -m src.extract

# ─── Treinamento LoRA (Python — CUDA/MPS/CPU) ────────────────────────────────
train: $(TRAIN_FILE)
	@echo "→ Iniciando fine-tuning LoRA (Python)..."
	@uv run python -m src.train

# ─── Exportação LoRA HF → GGUF ───────────────────────────────────────────────
export-lora: $(MODEL_LORA_HF)/adapter_model.safetensors
	@echo "→ Convertendo adaptador HF para GGUF..."
	@mkdir -p models/lora
	@uv run python $(LLAMA_DIR)/convert_lora_to_gguf.py \
		--base-model-id $(MODEL_HF_ID) \
		--outfile $(MODEL_LORA) \
		$(MODEL_LORA_HF)
	@echo "✓ Adaptador GGUF em: $(MODEL_LORA)"

# ─── Merge LoRA + push para Hugging Face Hub ─────────────────────────────────
push-hf: $(MODEL_LORA_HF)/adapter_model.safetensors
	@echo "→ Fundindo adaptador LoRA no modelo base e publicando no HF Hub..."
	@uv run python -m src.merge

# ─── Inferência interativa ────────────────────────────────────────────────────
run: $(MODEL_GGUF) $(MODEL_LORA)
	@echo "→ Iniciando chat interativo (llama-cli + LoRA GGUF)..."
	@$(LLAMA_BIN)/llama-cli \
		--model $(MODEL_GGUF) \
		--lora $(MODEL_LORA) \
		--interactive \
		--color \
		-c 4096 \
		-t $(CPU_THREADS) \
		-sys "$(shell uv run python -c "import yaml; print(yaml.safe_load(open('prompts/prompts.yaml'))['system'].strip())")"

# ─── Limpeza ──────────────────────────────────────────────────────────────────
clean:
	@echo "→ Removendo dados processados e adaptadores LoRA..."
	@rm -rf $(DATA_DIR)/* models/lora/* $(MODEL_LORA_HF)/* $(MODEL_MERGED)
	@echo "✓ Limpeza concluída."

# ─── Guardas de arquivo ───────────────────────────────────────────────────────
$(TRAIN_FILE):
	@echo "⚠  Dataset não encontrado. Execute: make extract"
	@exit 1

$(MODEL_GGUF):
	@echo "⚠  Modelo GGUF não encontrado. Execute: make download-base"
	@exit 1

$(MODEL_LORA):
	@echo "⚠  Adaptador LoRA GGUF não encontrado. Execute: make export-lora"
	@exit 1

$(MODEL_LORA_HF)/adapter_model.safetensors:
	@echo "⚠  Adaptador HF não encontrado. Execute: make train"
	@exit 1

# ─── Docker ───────────────────────────────────────────────────────────────────
DOCKER_IMAGE := lora-smith-qwen
DOCKER_RUN   := docker run --rm --gpus all \
	-v $(PWD)/models:/app/models \
	-v $(PWD)/data:/app/data \
	-v $(PWD)/prompts:/app/prompts \
	--env-file .env \
	$(DOCKER_IMAGE)

docker-build:
	@echo "→ Construindo imagem Docker $(DOCKER_IMAGE)..."
	@docker build -t $(DOCKER_IMAGE) .
	@echo "✓ Imagem pronta: $(DOCKER_IMAGE)"

docker-extract:
	@echo "→ Extraindo posts dentro do container..."
	@$(DOCKER_RUN) make extract

docker-train:
	@echo "→ Iniciando fine-tuning dentro do container (CUDA)..."
	@$(DOCKER_RUN) make train

docker-export-lora:
	@echo "→ Convertendo adaptador HF → GGUF no container..."
	@$(DOCKER_RUN) make export-lora

docker-push-hf:
	@echo "→ Merge LoRA + push para HF Hub no container..."
	@$(DOCKER_RUN) make push-hf

docker-run:
	@echo "→ Chat interativo no container..."
	@docker run --rm -it --gpus all \
		-v $(PWD)/models:/app/models \
		--env-file .env \
		$(DOCKER_IMAGE) make run
