export type MetricsPayload = {
  architecture: {
    precision: string
    total_params_m: number
    active_params_m: number
    active_percentage: number
  }
  performance: {
    ttft_s: number
    tpot_s: number
    prefill_tps: number
    prefill_tflops: number
    decode_tps: number
    decode_tflops: number
  }
  memory: {
    framework: string
    active_mem_mb: number
    peak_mem_mb: number
  }
}

export type InferenceStatusPayload = {
  phase: string
  message: string
  model_loaded: boolean
  backend: string
  last_error?: string | null
}

export type MonitorPayload = {
  framework: string
  gpu_source: string
  chip_name?: string
  gpu_cores?: number
  gpu_note?: string
  gpu_util: number
  cpu_percent: number
  vram_used_gb: number
  vram_total_gb: number
  gpu_freq_mhz_distribution?: Record<string, number>
  ram_percent: number
  ram_used_gb: number
  ram_total_gb: number
  inference?: InferenceStatusPayload
}
