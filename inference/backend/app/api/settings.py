from fastapi import APIRouter

from app.core.state import app_state
from app.schemas.settings import InferenceSettings, SettingsResponse

router = APIRouter(prefix='/api', tags=['settings'])


@router.get('/setting', response_model=SettingsResponse)
@router.get('/settings', response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    return SettingsResponse(settings=app_state.get_settings())


@router.post('/setting', response_model=SettingsResponse)
@router.post('/settings', response_model=SettingsResponse)
def update_settings(settings: InferenceSettings) -> SettingsResponse:
    updated = app_state.update_settings(settings)
    return SettingsResponse(settings=updated)
