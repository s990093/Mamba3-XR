from fastapi import APIRouter

from app.schemas.metrics import MetricsPayload

router = APIRouter(prefix='/api/inf', tags=['inference'])


@router.post('/ana', response_model=MetricsPayload)
def analyze_model() -> MetricsPayload:
    return MetricsPayload(
        architecture={
            'precision': 'float16',
            'total_params_m': 546.1,
            'active_params_m': 234.08,
            'active_percentage': 42.86,
        },
        performance={
            'ttft_s': 0.082,
            'tpot_s': 0.031,
            'prefill_tps': 73.17,
            'prefill_tflops': 0.0343,
            'decode_tps': 32.35,
            'decode_tflops': 0.0151,
        },
        memory={
            'framework': 'MLX/CUDA',
            'active_mem_mb': 1243.48,
            'peak_mem_mb': 2741.41,
        },
    )
