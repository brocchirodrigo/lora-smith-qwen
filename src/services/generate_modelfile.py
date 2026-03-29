"""
Gera o Modelfile do Ollama com suporte nativo ao thinking mode do Qwen3.5.

O TEMPLATE prefill '<think>' no início de cada resposta do assistant,
forçando o modelo a raciocinar antes de responder — comportamento que
o fine-tuning já ensinou via dados de treino.

Uso:
    MODEL_MERGED_Q4=models/merged-q4km.gguf uv run python -m src.services.generate_modelfile
    # ou via Makefile:
    make ollama-create
"""

import os
import yaml

model = os.environ.get("MODEL_MERGED_Q4", "models/merged-q4km.gguf")
sys_prompt = yaml.safe_load(open("prompts/prompts.yaml"))["system"].strip()

# Template ChatML com prefill de <think> para ativar thinking mode.
# Ollama injeta este template a cada turno; ao iniciar a vez do assistant
# com '<think>\n', o modelo sempre começa gerando o bloco de raciocínio.
template = (
    "{{- if .System }}<|im_start|>system\n"
    "{{ .System }}<|im_end|>\n"
    "{{ end }}"
    "{{- range .Messages }}"
    '{{- if eq .Role "user" }}<|im_start|>user\n'
    "{{ .Content }}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n"
    '{{- else if eq .Role "assistant" }}{{ .Content }}<|im_end|>\n'
    "{{ end }}"
    "{{- end }}"
)

modelfile = (
    f'FROM ./{model}\n'
    f'\n'
    f'SYSTEM """{sys_prompt}"""\n'
    f'\n'
    f'TEMPLATE """{template}"""\n'
    f'\n'
    'PARAMETER num_ctx 4096\n'
    'PARAMETER num_predict 2048\n'
    'PARAMETER temperature 0.6\n'
    'PARAMETER top_p 0.95\n'
    'PARAMETER top_k 20\n'
    'PARAMETER min_p 0.0\n'
    'PARAMETER repeat_penalty 1.0\n'
    'PARAMETER stop "<|im_end|>"\n'
    'PARAMETER stop "<|im_start|>"\n'
    'PARAMETER stop "<|endoftext|>"\n'
)

output = os.environ.get("MODELFILE_OUT", "Modelfile")
with open(output, "w", encoding="utf-8") as f:
    f.write(modelfile)

print(f"  ✓ {output} gerado (modelo: {model}, thinking: ativo, temp: 0.6)")
