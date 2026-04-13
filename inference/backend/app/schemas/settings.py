from pydantic import BaseModel, Field


class InferenceSettings(BaseModel):
    top_k: int = Field(default=0, ge=0, le=200)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    rep_pen: float = Field(default=1.05, ge=0.5, le=2.0)
    pres_pen: float = Field(default=0.0, ge=0.0, le=2.0)
    freq_pen: float = Field(default=0.05, ge=0.0, le=2.0)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=1, le=4096)
    no_eos_stop: bool = Field(default=False)
    system_prompt: str = Field(default='You are a helpful assistant.')


class SettingsResponse(BaseModel):
    settings: InferenceSettings
