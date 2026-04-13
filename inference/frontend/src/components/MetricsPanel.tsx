import type { MetricsPayload } from '@/types/metrics'

type Props = {
  metrics: MetricsPayload
}

export function MetricsPanel({ metrics }: Props) {
  return (
    <div className='mt-4 grid grid-cols-1 gap-3 md:grid-cols-3'>
      <div className='rounded-xl border border-[#3d4a44]/40 bg-[#282935] p-4 text-xs'>
        <div className='mb-2 text-[10px] uppercase tracking-widest text-slate-500'>Architecture</div>
        <div className='flex justify-between py-1 text-slate-300'>
          <span>Precision</span>
          <span className='font-mono text-[#61dbb4]'>{metrics.architecture.precision}</span>
        </div>
        <div className='flex justify-between py-1 text-slate-300'>
          <span>Total Params</span>
          <span className='font-mono'>{metrics.architecture.total_params_m}M</span>
        </div>
        <div className='flex justify-between py-1 text-slate-300'>
          <span>Active Util</span>
          <span className='font-mono text-[#61dbb4]'>{metrics.architecture.active_percentage}%</span>
        </div>
      </div>
      <div className='rounded-xl border border-[#3d4a44]/40 bg-[#282935] p-4 text-xs'>
        <div className='mb-2 text-[10px] uppercase tracking-widest text-slate-500'>Latency</div>
        <div className='py-1 text-slate-300'>
          TTFT <span className='ml-2 font-mono text-base text-white'>{metrics.performance.ttft_s}s</span>
        </div>
        <div className='py-1 text-slate-300'>
          TPOT <span className='ml-2 font-mono text-base text-[#61dbb4]'>{metrics.performance.tpot_s}s/token</span>
        </div>
      </div>
      <div className='rounded-xl border border-[#3d4a44]/40 bg-[#282935] p-4 text-xs'>
        <div className='mb-2 text-[10px] uppercase tracking-widest text-slate-500'>Throughput</div>
        <div className='flex justify-between rounded bg-[#1a1b26] px-2 py-1 text-slate-300'>
          <span className='font-mono'>Prefill</span>
          <span className='font-mono'>{metrics.performance.prefill_tps} t/s</span>
        </div>
        <div className='mt-2 flex justify-between rounded bg-[#1a1b26] px-2 py-1 text-slate-300'>
          <span className='font-mono'>Decode</span>
          <span className='font-mono'>{metrics.performance.decode_tps} t/s</span>
        </div>
      </div>
    </div>
  )
}
