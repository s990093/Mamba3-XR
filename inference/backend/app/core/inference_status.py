from __future__ import annotations

from threading import Lock
from typing import TypedDict


class InferenceStatusSnapshot(TypedDict, total=False):
    phase: str
    message: str
    model_loaded: bool
    backend: str
    last_error: str | None


class InferenceStatus:
    """Thread-safe inference lifecycle for UI (load model, ready, generating, error)."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._phase = "idle"
        self._message = "Model loads on first prompt — send to start"
        self._model_loaded = False
        self._last_error: str | None = None

    def set_loading(self, message: str) -> None:
        with self._lock:
            self._phase = "loading_model"
            self._message = message
            self._last_error = None

    def set_ready(self, message: str = "Model ready") -> None:
        with self._lock:
            self._phase = "ready"
            self._message = message
            self._model_loaded = True
            self._last_error = None

    def set_generating(self, message: str = "Generating…") -> None:
        with self._lock:
            self._phase = "generating"
            self._message = message

    def set_error(self, err: str) -> None:
        with self._lock:
            self._phase = "error"
            self._message = err[:500]
            self._last_error = err[:500]

    def set_idle(self, message: str = "Idle") -> None:
        with self._lock:
            self._phase = "idle"
            self._message = message

    def snapshot(self, backend: str) -> InferenceStatusSnapshot:
        with self._lock:
            return {
                "phase": self._phase,
                "message": self._message,
                "model_loaded": self._model_loaded,
                "backend": backend,
                "last_error": self._last_error,
            }


inference_status = InferenceStatus()
