'use client'

import { useEffect, useMemo, useState } from 'react'
import { Settings, Trash2 } from 'lucide-react'

import { useInfStore } from '@/store/useInfStore'

type Props = {
  onOpenSettings: () => void
}

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000'
const HISTORY_POINTS = 60

function Sparkline({ values }: { values: number[] }) {
  const points = useMemo(() => {
    if (values.length === 0) return ''
    return values
      .map((v, i) => {
        const x = (i / Math.max(values.length - 1, 1)) * 100
        const y = 100 - Math.max(0, Math.min(100, v))
        return `${x},${y}`
      })
      .join(' ')
  }, [values])

  return (
    <svg viewBox='0 0 100 100' preserveAspectRatio='none' className='h-6 w-28'>
      <polyline
        points={points}
        fill='none'
        stroke='rgb(97 219 180)'
        strokeWidth='4'
        strokeLinecap='round'
        strokeLinejoin='round'
      />
    </svg>
  )
}

export function TopMonitorBar({ onOpenSettings }: Props) {
  const monitor = useInfStore((s) => s.monitor)
  const [gpuHistory, setGpuHistory] = useState<number[]>([])
  const [cpuHistory, setCpuHistory] = useState<number[]>([])
  const [ramHistory, setRamHistory] = useState<number[]>([])

  const flushMemory = async () => {
    await fetch(`${apiBase}/api/flush`, { method: 'POST' })
  }

  const usageColor = (value: number) => {
    if (value > 90) return 'bg-red-500'
    if (value > 70) return 'bg-yellow-500'
    return 'bg-emerald-500'
  }

  const inf = monitor?.inference
  const gpuFreqLabel = useMemo(() => {
    const dist = monitor?.gpu_freq_mhz_distribution
    if (!dist) return ''
    const entries = Object.entries(dist)
      .filter(([, percent]) => percent > 0)
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .slice(0, 6)
    if (entries.length === 0) return ''
    return entries.map(([mhz, percent]) => `${mhz}MHz ${percent.toFixed(1)}%`).join(' | ')
  }, [monitor?.gpu_freq_mhz_distribution])

  useEffect(() => {
    if (!monitor) return
    setGpuHistory((prev) => [...prev.slice(-(HISTORY_POINTS - 1)), monitor.gpu_util ?? 0])
    setCpuHistory((prev) => [...prev.slice(-(HISTORY_POINTS - 1)), monitor.cpu_percent ?? 0])
    setRamHistory((prev) => [...prev.slice(-(HISTORY_POINTS - 1)), monitor.ram_percent ?? 0])
  }, [monitor])

  const phaseLabel = (() => {
    switch (inf?.phase) {
      case 'loading_model':
        return '載入模型'
      case 'ready':
        return '就緒'
      case 'generating':
        return '生成中'
      case 'error':
        return '錯誤'
      default:
        return '待命'
    }
  })()

  const phaseClass = (() => {
    switch (inf?.phase) {
      case 'loading_model':
        return 'text-amber-300'
      case 'ready':
        return 'text-emerald-400'
      case 'generating':
        return 'text-cyan-300'
      case 'error':
        return 'text-red-400'
      default:
        return 'text-zinc-400'
    }
  })()

  return (
    <header className='fixed top-0 z-50 w-full border-b border-[#3d4a44]/30 bg-gradient-to-b from-[#1a1b26] to-[#11131d] px-4 py-3 shadow-[0_0_20px_rgba(97,219,180,0.06)]'>
      <div className='mx-auto flex max-w-7xl items-center justify-between gap-4'>
        <div className='flex items-center gap-4'>
          <div className='text-xl font-black tracking-tight text-[#61dbb4]'>Inf-Platform</div>
          <div className='hidden h-4 w-px bg-[#3d4a44] md:block' />
          <div className='hidden min-w-0 md:block'>
            <div className='text-[10px] uppercase tracking-widest text-slate-500'>Model</div>
            <div className='text-sm text-zinc-200'>Llama-3-8B-Instruct (fp16)</div>
            <div className='mt-1 flex min-w-0 flex-col gap-0.5'>
              <div className='flex flex-wrap items-center gap-2 text-[11px]'>
                <span className='rounded border border-zinc-600/80 px-1.5 py-0.5 text-[10px] uppercase tracking-wide text-slate-500'>
                  狀態
                </span>
                <span className={`font-medium ${phaseClass}`}>{phaseLabel}</span>
                <span className='truncate text-zinc-500' title={inf?.message ?? ''}>
                  {inf?.message ?? '—'}
                </span>
              </div>
              {inf?.backend ? (
                <span className='text-[10px] text-zinc-600'>backend: {inf.backend}</span>
              ) : null}
            </div>
          </div>
          <div className='flex min-w-0 flex-col items-end gap-0.5 md:hidden'>
            <span className='rounded border border-zinc-700 px-1.5 py-0.5 text-[11px] text-zinc-300'>
              GPU {monitor?.gpu_util ?? 0}%
            </span>
            <span className={`max-w-[52vw] truncate text-[10px] font-medium ${phaseClass}`} title={inf?.message}>
              {phaseLabel}
              {inf?.message ? ` · ${inf.message}` : ''}
            </span>
          </div>
        </div>
        <div className='hidden min-w-0 flex-1 items-center justify-center gap-6 text-xs text-zinc-300 md:flex'>
          <div className='w-44'>
            <div className='mb-1 flex justify-between text-[10px] uppercase tracking-widest'>
              <span className='text-slate-400'>GPU Util</span>
              <span className='text-[#61dbb4]'>{monitor?.gpu_util ?? 0}%</span>
            </div>
            <div className='h-1.5 rounded bg-zinc-700'>
              <div
                className={`h-1.5 rounded transition-all ${usageColor(monitor?.gpu_util ?? 0)}`}
                style={{ width: `${Math.min(monitor?.gpu_util ?? 0, 100)}%` }}
              />
            </div>
            <div className='mt-1 opacity-80'>
              <Sparkline values={gpuHistory} />
            </div>
          </div>
          <div className='w-44'>
            <div className='mb-1 flex justify-between text-[10px] uppercase tracking-widest'>
              <span className='text-slate-400'>CPU Load</span>
              <span className='text-[#61dbb4]'>{monitor?.cpu_percent ?? 0}%</span>
            </div>
            <div className='h-1.5 rounded bg-zinc-700'>
              <div
                className={`h-1.5 rounded transition-all ${usageColor(monitor?.cpu_percent ?? 0)}`}
                style={{ width: `${Math.min(monitor?.cpu_percent ?? 0, 100)}%` }}
              />
            </div>
            <div className='mt-1 opacity-80'>
              <Sparkline values={cpuHistory} />
            </div>
          </div>
          <div className='w-44'>
            <div className='mb-1 flex justify-between text-[10px] uppercase tracking-widest'>
              <span className='text-slate-400'>RAM</span>
              <span className='text-[#61dbb4]'>{monitor?.ram_percent ?? 0}%</span>
            </div>
            <div className='h-1.5 rounded bg-zinc-700'>
              <div
                className={`h-1.5 rounded transition-all ${usageColor(monitor?.ram_percent ?? 0)}`}
                style={{ width: `${Math.min(monitor?.ram_percent ?? 0, 100)}%` }}
              />
            </div>
            <div className='mt-1 opacity-80'>
              <Sparkline values={ramHistory} />
            </div>
          </div>
          <span className='truncate text-zinc-400'>
            VRAM {monitor?.vram_used_gb ?? 0} / {monitor?.vram_total_gb ?? 0} GB
            {monitor?.gpu_source ? ` (${monitor.gpu_source})` : ''}
          </span>
          {monitor?.chip_name ? (
            <span className='truncate text-[10px] text-zinc-400'>
              {monitor.chip_name}
              {monitor?.gpu_cores ? ` · GPU Cores ${monitor.gpu_cores}` : ''}
            </span>
          ) : null}
          {gpuFreqLabel ? (
            <span className='truncate text-[10px] text-cyan-300' title={gpuFreqLabel}>
              Freq: {gpuFreqLabel}
            </span>
          ) : null}
          {monitor?.gpu_note ? <span className='truncate text-[10px] text-amber-300'>{monitor.gpu_note}</span> : null}
        </div>
        <div className='flex items-center gap-2'>
          <button
            onClick={flushMemory}
            className='flex items-center gap-1 rounded-md border border-zinc-700 px-2 py-1 text-xs text-zinc-100 hover:bg-zinc-800'
          >
            <Trash2 size={14} /> Flush
          </button>
          <button
            onClick={onOpenSettings}
            className='rounded-md border border-zinc-700 p-1 text-zinc-100 hover:bg-zinc-800'
          >
            <Settings size={16} />
          </button>
        </div>
      </div>
    </header>
  )
}
