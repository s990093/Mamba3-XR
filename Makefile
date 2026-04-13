# Mamba3-XR — local dev shortcuts (run from repo root: `make mlx-bench`)
#
# Override examples:
#   make mlx-bench CHECKPOINT=checkpoint.pt
#   make mlx-bench SEQ_LEN=256 DECODE_TOK=64
#   make mlx-bench BENCH_EXTRA='--prompt "Hi"'   # no truncation unless SEQ_LEN is set
#   make mlx-export-npz CHECKPOINT=weights/checkpoint.pt

ROOT := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

# Prefer repo .venv so `make mlx-bench` works without activating the venv
VENV_PY := $(ROOT)/.venv/bin/python3
PYTHON ?= $(if $(wildcard $(VENV_PY)),$(VENV_PY),python3)
BENCH := $(ROOT)/inference/benchmark_mlx.py
STREAM := $(ROOT)/inference/stream_mlx.py
PROF := $(ROOT)/inference/profile_mlx_infer.py
TOK ?= $(ROOT)/inference/tokenizer
PROFILE_DECODE_STEPS ?= 32
# Preset: throughput | safe | eager | sequential-ssm | custom
INFER_TYPE ?= throughput
# Model compute / weight dtype: fp32 | bf16 | fp16
DTYPE ?= bf16

# Leave empty to use resolve_mlx_checkpoint(): repo model.npz → checkpoint.pt sidecars
CHECKPOINT ?=

# Optional max prefill tokens. Empty = use prompt token length directly (no truncation).
SEQ_LEN ?=
DECODE_TOK ?= 128
WARMUP ?= 2
KV_DTYPE ?= bf16
ROUTER_TEMP ?= 0.5

# Non-empty CHECKPOINT → " --checkpoint path" (leading space; empty when unset)
CKPT_ARG = $(if $(strip $(CHECKPOINT)), --checkpoint $(CHECKPOINT),)
# Non-empty SEQ_LEN → " --seq-len N"
SEQ_ARG = $(if $(strip $(SEQ_LEN)), --seq-len $(SEQ_LEN),)

# Extra benchmark args, e.g. BENCH_EXTRA='--prompt "Hello" --decode-tokens 64' or --no-show-io
BENCH_EXTRA ?=

.PHONY: help mlx-bench mlx-bench-quick mlx-stream mlx-profile mlx-export-npz mlx-force-pt deps-mlx frontend-dev frontend backend-dev backend up

help:
	@echo "Mamba3-XR Makefile"
	@echo ""
	@echo "  make mlx-bench          MLX prefill/decode benchmark (default tokenizer + seq)"
	@echo "  make mlx-bench-quick    Shorter run (SEQ_LEN=128, DECODE_TOK=32)"
	@echo "  make mlx-stream         Stream tokens to stdout (default includes full compile + 4-bit quant)"
	@echo "  make mlx-export-npz     Load .pt, write .npz cache next to checkpoint (set CHECKPOINT=...)"
	@echo "  make mlx-force-pt       Same as mlx-bench but --force-pt"
	@echo "  make mlx-profile        Layer/host-GPU proxy profiler (see inference/profile_mlx_infer.py)"
	@echo "  make backend-dev        Start FastAPI backend with --reload"
	@echo "  make backend            Start FastAPI backend (production mode)"
	@echo "  make frontend-dev       Start Next.js frontend with hot-reload"
	@echo "  make frontend           Start Next.js frontend (production mode)"
	@echo "  make up                Start backend-dev + frontend-dev together"
	@echo "  make deps-mlx           pip install mlx numpy transformers torch"
	@echo ""
	@echo "Variables: CHECKPOINT, SEQ_LEN(optional), DECODE_TOK, MAX_NEW_TOK, WARMUP, DTYPE, KV_DTYPE, ROUTER_TEMP, INFER_TYPE, BENCH_EXTRA, STREAM_EXTRA, STREAM_QUANT, BACKEND_HOST, BACKEND_PORT, BACKEND_EXTRA, FRONTEND_PORT, FRONTEND_EXTRA, FRONTEND_API_BASE, FRONTEND_WS_BASE, PYTHON"
	@echo "Example:   make mlx-bench CHECKPOINT=checkpoint.pt SEQ_LEN=1024"
	@echo "Example:   make mlx-bench BENCH_EXTRA='--prompt \"Hi\" --decode-tokens 32'"

# Core benchmark (auto model.npz / checkpoint.pt when CHECKPOINT is empty)
mlx-bench:
	$(PYTHON) $(BENCH)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE)$(SEQ_ARG) --decode-tokens $(DECODE_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP) $(BENCH_EXTRA) 

mlx-bench-quick:
	$(MAKE) mlx-bench SEQ_LEN=128 DECODE_TOK=512 WARMUP=2

# Streaming generation (same checkpoint/tokenizer vars as mlx-bench; MAX_NEW_TOK replaces decode length)
MAX_NEW_TOK ?= 2048
# Default stream mode: full decode compile + continue on EOS + 4-bit quant for speed.
STREAM_QUANT ?= 4
STREAM_QUANT_ARG = $(if $(strip $(STREAM_QUANT)), --quantize $(STREAM_QUANT),)
STREAM_EXTRA ?= --full-decode-compile

mlx-stream:
	$(PYTHON) $(STREAM)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE)$(SEQ_ARG) --max-new-tokens $(MAX_NEW_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP)$(STREAM_QUANT_ARG) $(STREAM_EXTRA) --no-eos-stop

# Bottleneck report: wall vs thread CPU, MLX peak memory (does not modify mlx_hybrid_infer.py)
mlx-profile:
	$(PYTHON) $(PROF)$(CKPT_ARG) --tokenizer $(TOK) --dtype $(DTYPE) --kv-dtype $(KV_DTYPE)$(SEQ_ARG) --profile-decode-steps $(PROFILE_DECODE_STEPS) $(BENCH_EXTRA)

mlx-force-pt:
	$(PYTHON) $(BENCH)$(CKPT_ARG) --tokenizer $(TOK) --inference-type $(INFER_TYPE) --dtype $(DTYPE) --force-pt$(SEQ_ARG) --decode-tokens $(DECODE_TOK) --warmup $(WARMUP) --kv-dtype $(KV_DTYPE) --router-temp $(ROUTER_TEMP) $(BENCH_EXTRA)

# After success, next `make mlx-bench` can load the .npz without torch
mlx-export-npz:
	@test -n "$(CHECKPOINT)" || (echo "Set CHECKPOINT=path/to/model.pt" && exit 1)
	$(PYTHON) $(BENCH) \
		--checkpoint $(CHECKPOINT) \
		--tokenizer $(TOK) \
		--force-pt \
		$(SEQ_ARG) \
		--decode-tokens $(DECODE_TOK) \
		--save-npz

deps-mlx:
	$(PYTHON) -m pip install -U mlx numpy transformers torch

# Backend (FastAPI)
BACKEND_DIR := $(ROOT)/inference/backend
BACKEND_HOST ?= 0.0.0.0
BACKEND_PORT ?= 8000
BACKEND_EXTRA ?=
BACKEND_NO_EOS_STOP ?= 0
FRONTEND_DIR := $(ROOT)/inference/frontend
FRONTEND_PORT ?= 3000
FRONTEND_EXTRA ?=
FRONTEND_API_BASE ?= http://localhost:$(BACKEND_PORT)
FRONTEND_WS_BASE ?= ws://localhost:$(BACKEND_PORT)

backend-dev:
	@if [ ! -d "$(BACKEND_DIR)" ]; then echo "Backend dir not found: $(BACKEND_DIR)"; exit 1; fi
	@if [ ! -f "$(BACKEND_DIR)/.env" ] && [ -f "$(BACKEND_DIR)/.env.example" ]; then cp "$(BACKEND_DIR)/.env.example" "$(BACKEND_DIR)/.env"; fi
	cd "$(BACKEND_DIR)" && INFERENCE_NO_EOS_STOP="$(BACKEND_NO_EOS_STOP)" $(PYTHON) -m uvicorn app.main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) --reload $(BACKEND_EXTRA)

backend:
	@if [ ! -d "$(BACKEND_DIR)" ]; then echo "Backend dir not found: $(BACKEND_DIR)"; exit 1; fi
	@if [ ! -f "$(BACKEND_DIR)/.env" ] && [ -f "$(BACKEND_DIR)/.env.example" ]; then cp "$(BACKEND_DIR)/.env.example" "$(BACKEND_DIR)/.env"; fi
	cd "$(BACKEND_DIR)" && INFERENCE_NO_EOS_STOP="$(BACKEND_NO_EOS_STOP)" $(PYTHON) -m uvicorn app.main:app --host $(BACKEND_HOST) --port $(BACKEND_PORT) $(BACKEND_EXTRA)

frontend-dev:
	@if [ ! -d "$(FRONTEND_DIR)" ]; then echo "Frontend dir not found: $(FRONTEND_DIR)"; exit 1; fi
	cd "$(FRONTEND_DIR)" && NEXT_PUBLIC_API_BASE="$(FRONTEND_API_BASE)" NEXT_PUBLIC_WS_BASE="$(FRONTEND_WS_BASE)" npm run dev -- --port $(FRONTEND_PORT) $(FRONTEND_EXTRA)

frontend:
	@if [ ! -d "$(FRONTEND_DIR)" ]; then echo "Frontend dir not found: $(FRONTEND_DIR)"; exit 1; fi
	cd "$(FRONTEND_DIR)" && NEXT_PUBLIC_API_BASE="$(FRONTEND_API_BASE)" NEXT_PUBLIC_WS_BASE="$(FRONTEND_WS_BASE)" npm run start -- --port $(FRONTEND_PORT) $(FRONTEND_EXTRA)

up:
	@if [ ! -d "$(BACKEND_DIR)" ]; then echo "Backend dir not found: $(BACKEND_DIR)"; exit 1; fi
	@if [ ! -d "$(FRONTEND_DIR)" ]; then echo "Frontend dir not found: $(FRONTEND_DIR)"; exit 1; fi
	@if [ ! -f "$(BACKEND_DIR)/.env" ] && [ -f "$(BACKEND_DIR)/.env.example" ]; then cp "$(BACKEND_DIR)/.env.example" "$(BACKEND_DIR)/.env"; fi
	@bash -lc 'set -e; trap "kill 0" INT TERM EXIT; \
		cd "$(BACKEND_DIR)" && INFERENCE_NO_EOS_STOP="$(BACKEND_NO_EOS_STOP)" "$(PYTHON)" -m uvicorn app.main:app --host "$(BACKEND_HOST)" --port "$(BACKEND_PORT)" --reload $(BACKEND_EXTRA) & \
		cd "$(FRONTEND_DIR)" && NEXT_PUBLIC_API_BASE="$(FRONTEND_API_BASE)" NEXT_PUBLIC_WS_BASE="$(FRONTEND_WS_BASE)" npm run dev -- --port "$(FRONTEND_PORT)" $(FRONTEND_EXTRA)'
