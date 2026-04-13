# Backend

FastAPI backend for Inf-Platform.

## Setup

```bash
# Use your existing virtual environment
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Environment

```bash
cp .env.example .env
```

- `INFERENCE_BACKEND=inf` (default): bridge to `inf/inf/main.py` (your primary Mamba3 pipeline)
- `INFERENCE_BACKEND=mock`: lightweight streaming mock for UI/dev
