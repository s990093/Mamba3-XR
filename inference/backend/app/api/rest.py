from fastapi import APIRouter

from app.core.state import app_state
from app.utils.hardware import full_monitor_payload

router = APIRouter(prefix='/api', tags=['state'])


@router.get('/status')
def api_status() -> dict[str, object]:
    """Same payload as WebSocket `monitor` messages (hardware + inference phase)."""
    return full_monitor_payload()


@router.post('/rest')
def reset_chat_state() -> dict[str, str]:
    app_state.reset()
    return {'status': 'ok', 'message': 'Chat context reset'}
