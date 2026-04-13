from pydantic import BaseModel


class ArchitectureMetrics(BaseModel):
    precision: str
    total_params_m: float
    active_params_m: float
    active_percentage: float


class PerformanceMetrics(BaseModel):
    ttft_s: float
    tpot_s: float
    prefill_tps: float
    prefill_tflops: float
    decode_tps: float
    decode_tflops: float


class MemoryMetrics(BaseModel):
    framework: str
    active_mem_mb: float
    peak_mem_mb: float


class MetricsPayload(BaseModel):
    architecture: ArchitectureMetrics
    performance: PerformanceMetrics
    memory: MemoryMetrics
