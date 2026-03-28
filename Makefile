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
MODEL_HF_ID   ?= Qwen/Qwen3.5-2B
MODEL_REPO_ID ?= unsloth/Qwen3.5-2B-GGUF
MODEL_FILENAME ?= Qwen3.5-2B-Q4_K_M.gguf

MODEL_MERGED      := models/merged
MODEL_MERGED_F16  := models/merged-f16.gguf
MODEL_MERGED_Q4   := models/merged-q4km.gguf
HF_PUSH_REPO  ?=
HF_TOKEN      ?=
OLLAMA_MODEL  ?=

MODEL_GGUF := $(MODEL_BASE)/$(MODEL_FILENAME)

.PHONY: help setup download-base extract train merge export export-lora export-merged-gguf push push-hf push-gguf ollama-create push-ollama run clean \
        docker-build docker-extract docker-train docker-export docker-export-lora docker-push docker-push-hf docker-run

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
	@echo "    make merge              Funde LoRA no modelo base (salva local em models/merged)"
	@echo "    make export             export-lora + merge + export-merged-gguf"
	@echo "    make push               Publica safetensors + GGUF no Hugging Face Hub"
	@echo "    make push-hf            Apenas safetensors (requer make merge)"
	@echo "    make push-gguf          Apenas GGUF (requer make export)"
	@echo "    make ollama-create      Registra modelo no Ollama local (requer make export)"
	@echo "    make push-ollama        Publica modelo no ollama.com"
	@echo "    make run                Chat interativo com llama-cli + LoRA GGUF"
	@echo "    make clean              Remove dados processados e adaptadores"
	@echo ""
	@echo "  Comandos Docker (Linux + CUDA):"
	@echo "    make docker-build       Constrói a imagem Docker"
	@echo "    make docker-extract     Extrai posts dentro do container"
	@echo "    make docker-train       Fine-tuning dentro do container (GPU)"
	@echo "    make docker-export      export-lora + merge + export-merged-gguf no container"
	@echo "    make docker-push        Publica safetensors + GGUF no HF Hub no container"
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
repo = os.environ.get('MODEL_REPO_ID', 'unsloth/Qwen3.5-2B-GGUF'); \
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

# ─── Merge LoRA no modelo base (local) ───────────────────────────────────────
merge: $(MODEL_LORA_HF)/adapter_model.safetensors
	@echo "→ Fundindo adaptador LoRA no modelo base..."
	@uv run python -m src.merge merge

# ─── Export: adapter GGUF (llama-cli) + merge local + GGUF fundido (LM Studio)
export: $(MODEL_LORA_HF)/adapter_model.safetensors
	@$(MAKE) --no-print-directory export-lora
	@$(MAKE) --no-print-directory merge
	@$(MAKE) --no-print-directory export-merged-gguf

# ─── Push safetensors para o Hugging Face Hub ────────────────────────────────
push-hf: $(MODEL_MERGED)/config.json
	@echo "→ Publicando safetensors no HF Hub..."
	@uv run python -m src.merge push

# ─── Push GGUF + safetensors para o Hugging Face Hub ─────────────────────────
push: $(MODEL_MERGED)/config.json $(MODEL_MERGED_Q4)
	@$(MAKE) --no-print-directory push-hf
	@$(MAKE) --no-print-directory push-gguf
	@$(MAKE) --no-print-directory push-ollama

# ─── Inferência interativa ────────────────────────────────────────────────────
run: $(MODEL_GGUF) $(MODEL_LORA)
	@echo "→ Iniciando chat interativo (llama-cli + LoRA GGUF)..."
	@SYS=$$(uv run python -c "import yaml; print(yaml.safe_load(open('prompts/prompts.yaml'))['system'].strip())") && \
	$(LLAMA_BIN)/llama-cli \
		--model $(MODEL_GGUF) \
		--lora $(MODEL_LORA) \
		--conversation \
		--color on \
		-c 4096 \
		-t $(CPU_THREADS) \
		-sys "$$SYS" \
		--prompt "<|im_start|>assistant\nOlá, como posso te ajudar?<|im_end|>"

# ─── Export GGUF do modelo fundido (para LM Studio) ─────────────────────────
export-merged-gguf: $(MODEL_MERGED)/config.json
	@echo "→ Convertendo modelo fundido para GGUF F16..."
	@uv run python $(LLAMA_DIR)/convert_hf_to_gguf.py \
		$(MODEL_MERGED) \
		--outfile $(MODEL_MERGED_F16) \
		--outtype f16
	@echo "→ Quantizando para Q4_K_M..."
	@$(LLAMA_DIR)/build/tools/quantize $(MODEL_MERGED_F16) $(MODEL_MERGED_Q4) Q4_K_M
	@rm -f $(MODEL_MERGED_F16)
	@echo "✓ GGUF em: $(MODEL_MERGED_Q4)"

push-gguf: $(MODEL_MERGED_Q4)
	@echo "→ Publicando GGUF no Hugging Face Hub: $(HF_PUSH_REPO)..."
	@uv run python -c "\
from huggingface_hub import HfApi; \
import os; \
api = HfApi(); \
fname = '$(notdir $(MODEL_MERGED_Q4))'; \
api.upload_file( \
    path_or_fileobj='$(MODEL_MERGED_Q4)', \
    path_in_repo=fname, \
    repo_id='$(HF_PUSH_REPO)', \
    token='$(HF_TOKEN)', \
); \
print(f'✓ Disponível em: https://huggingface.co/$(HF_PUSH_REPO)/blob/main/{fname}')"

# ─── Ollama ───────────────────────────────────────────────────────────────────
ollama-create: $(MODEL_MERGED_Q4)
	@if [ -z "$(OLLAMA_MODEL)" ]; then echo "  → OLLAMA_MODEL não definido, pulando ollama-create."; exit 0; fi
	@if ! command -v ollama >/dev/null 2>&1; then echo "  → ollama não instalado, pulando ollama-create."; exit 0; fi
	@echo "→ Gerando Modelfile..."
	@SYS=$$(uv run python -c "import yaml; print(yaml.safe_load(open('prompts/prompts.yaml'))['system'].strip())") && \
	printf 'FROM ./%s\n\nSYSTEM """%s"""\n\nPARAMETER num_ctx 4096\nPARAMETER temperature 0.7\n' \
		"$(MODEL_MERGED_Q4)" "$$SYS" > Modelfile
	@echo "→ Registrando modelo no Ollama local: $(OLLAMA_MODEL)..."
	@ollama create $(OLLAMA_MODEL) -f Modelfile
	@echo "✓ Pronto. Para testar: ollama run $(OLLAMA_MODEL)"

push-ollama:
	@if [ -z "$(OLLAMA_MODEL)" ]; then echo "  → OLLAMA_MODEL não definido, pulando push-ollama."; exit 0; fi
	@if [ -z "$(OLLAMA_API_KEY)" ]; then echo "  → OLLAMA_API_KEY não definida, pulando push-ollama."; exit 0; fi
	@if ! command -v ollama >/dev/null 2>&1; then echo "  → ollama não instalado, pulando push-ollama."; exit 0; fi
	@echo "→ Publicando no Ollama: $(OLLAMA_MODEL)..."
	@OLLAMA_API_KEY=$(OLLAMA_API_KEY) ollama push $(OLLAMA_MODEL)
	@echo "✓ Disponível em: https://ollama.com/$(OLLAMA_MODEL)"

# ─── Limpeza ──────────────────────────────────────────────────────────────────
clean:
	@echo "→ Removendo dados, adaptadores, modelos fundidos e cache HuggingFace..."
	@rm -rf $(DATA_DIR)/* models/lora/* $(MODEL_LORA_HF)/* $(MODEL_MERGED) $(MODEL_MERGED_Q4) $(MODEL_BASE)/*
	@huggingface-cli delete-cache --yes 2>/dev/null || true
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

$(MODEL_MERGED)/config.json:
	@echo "⚠  Modelo fundido não encontrado. Execute: make merge"
	@exit 1

$(MODEL_MERGED_Q4):
	@echo "⚠  GGUF fundido não encontrado. Execute: make export"
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

docker-export:
	@echo "→ Export completo (adapter GGUF + merge + GGUF fundido) no container..."
	@$(DOCKER_RUN) make export

docker-push:
	@echo "→ Publicando safetensors + GGUF no HF Hub no container..."
	@$(DOCKER_RUN) make push

docker-export-lora:
	@$(DOCKER_RUN) make export-lora

docker-push-hf:
	@$(DOCKER_RUN) make push-hf

docker-run:
	@echo "→ Chat interativo no container..."
	@docker run --rm -it --gpus all \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/prompts:/app/prompts \
		--env-file .env \
		$(DOCKER_IMAGE) make run
