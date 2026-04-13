'use client'

import { useEffect, useState } from 'react'

import { useInfStore } from '@/store/useInfStore'

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000'

type Props = {
  open: boolean
  onClose: () => void
}

export function SettingsModal({ open, onClose }: Props) {
  const settings = useInfStore((s) => s.settings)
  const setSettings = useInfStore((s) => s.setSettings)
  const [local, setLocal] = useState(settings)

  useEffect(() => setLocal(settings), [settings])
  useEffect(() => {
    if (!open) return
    ;(async () => {
      const res = await fetch(`${apiBase}/api/setting`)
      const data = await res.json()
      if (data.settings) {
        setSettings(data.settings)
        setLocal(data.settings)
      }
    })()
  }, [open, setSettings])

  if (!open) return null

  const save = async () => {
    const res = await fetch(`${apiBase}/api/setting`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(local),
    })
    const data = await res.json()
    if (data.settings) setSettings(data.settings)
    onClose()
  }

  return (
    <div className='fixed inset-0 z-50 flex items-center justify-center bg-black/50'>
      <div className='w-full max-w-md rounded-lg border border-zinc-700 bg-zinc-900 p-4 text-zinc-100'>
        <h2 className='mb-3 text-lg font-semibold'>Settings</h2>
        <div className='space-y-3 text-sm'>
          <label className='block'>
            Temperature
            <input
              type='range'
              min='0'
              max='2'
              step='0.01'
              value={local.temperature}
              onChange={(e) => setLocal({ ...local, temperature: Number(e.target.value) })}
              className='w-full'
            />
          </label>
          <label className='block'>
            Top K
            <input
              type='number'
              value={local.top_k}
              onChange={(e) => setLocal({ ...local, top_k: Number(e.target.value) })}
              className='mt-1 w-full rounded border border-zinc-600 bg-zinc-800 px-2 py-1'
            />
          </label>
          <label className='block'>
            Top P
            <input
              type='number'
              step='0.01'
              value={local.top_p}
              onChange={(e) => setLocal({ ...local, top_p: Number(e.target.value) })}
              className='mt-1 w-full rounded border border-zinc-600 bg-zinc-800 px-2 py-1'
            />
          </label>
          <label className='block'>
            Max Tokens
            <input
              type='number'
              value={local.max_tokens}
              onChange={(e) => setLocal({ ...local, max_tokens: Number(e.target.value) })}
              className='mt-1 w-full rounded border border-zinc-600 bg-zinc-800 px-2 py-1'
            />
          </label>
          <label className='flex items-center gap-2'>
            <input
              type='checkbox'
              checked={local.no_eos_stop}
              onChange={(e) => setLocal({ ...local, no_eos_stop: e.target.checked })}
            />
            No EOS Stop (ignore EOS and keep generating)
          </label>
          <label className='block'>
            System Prompt
            <textarea
              value={local.system_prompt}
              onChange={(e) => setLocal({ ...local, system_prompt: e.target.value })}
              className='mt-1 w-full rounded border border-zinc-600 bg-zinc-800 px-2 py-1'
            />
          </label>
        </div>
        <div className='mt-4 flex justify-end gap-2'>
          <button onClick={onClose} className='rounded border border-zinc-600 px-3 py-1 text-sm'>Cancel</button>
          <button onClick={save} className='rounded bg-emerald-600 px-3 py-1 text-sm text-white'>Save</button>
        </div>
      </div>
    </div>
  )
}
