import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.flush import router as flush_router
from app.api.inf import router as inf_router
from app.api.rest import router as rest_router
from app.api.settings import router as settings_router
from app.api.ws import router as ws_router
from app.core.inference import warmup_runtime_async
from app.core.inference_status import inference_status

load_dotenv()

# macOS: avoid noisy stderr when libs toggle malloc stack logging
for _k in ('MallocStackLogging', 'MallocStackLoggingNoCompact'):
    os.environ.pop(_k, None)

app = FastAPI(title='Inf-Platform Backend', version='0.1.0')

allowed_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:3001').split(',')
allow_origin_regex = os.getenv('CORS_ALLOW_ORIGIN_REGEX', r'^https?://(localhost|127\.0\.0\.1)(:\d+)?$')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in allowed_origins if o.strip()],
    allow_origin_regex=allow_origin_regex,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

app.include_router(settings_router)
app.include_router(rest_router)
app.include_router(flush_router)
app.include_router(inf_router)
app.include_router(ws_router)


@app.on_event('startup')
def preload_model_on_startup() -> None:
    if os.getenv('INFERENCE_PRELOAD_ON_STARTUP', '1').strip() != '1':
        return
    inference_status.set_loading('Startup preload: loading model in background…')
    warmup_runtime_async()


@app.get('/health')
def health_check() -> dict[str, str]:
    return {'status': 'ok'}
