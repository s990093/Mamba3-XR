from fastapi import APIRouter

from app.utils.memory import flush_memory

router = APIRouter(prefix='/api', tags=['memory'])


@router.post('/flush')
def flush_memory_api() -> dict[str, object]:
    return flush_memory()
