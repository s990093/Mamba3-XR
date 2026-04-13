from threading import Lock

from app.schemas.settings import InferenceSettings


class AppState:
    def __init__(self) -> None:
        self._lock = Lock()
        self._settings = InferenceSettings()
        self._chat_history: list[dict[str, str]] = []

    def get_settings(self) -> InferenceSettings:
        with self._lock:
            return self._settings

    def update_settings(self, settings: InferenceSettings) -> InferenceSettings:
        with self._lock:
            self._settings = settings
            return self._settings

    def append_message(self, role: str, content: str) -> None:
        with self._lock:
            self._chat_history.append({'role': role, 'content': content})

    def reset(self) -> None:
        with self._lock:
            self._chat_history.clear()


app_state = AppState()
