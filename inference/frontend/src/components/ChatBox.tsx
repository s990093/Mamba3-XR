'use client'

import { useState } from 'react'

import { useInferenceSocket } from '@/hooks/useInferenceSocket'
import { usePrefillScan } from '@/hooks/usePrefillScan'
import { useInfStore } from '@/store/useInfStore'
import { MetricsPanel } from './MetricsPanel'

const PREFILL_WINDOW = 10

function UserPromptWithPrefillCaret({
  content,
  scanIdx,
  headActive,
  hold,
}: {
  content: string
  scanIdx: number | null
  headActive: boolean
  hold: boolean
}) {
  if (scanIdx == null) {
    return <>{content}</>
  }
  const i = Math.min(Math.max(0, scanIdx), content.length)
  if (i === 0) {
    return <>{content}</>
  }

  const winStart = Math.max(0, i - PREFILL_WINDOW)
  const before = content.slice(0, winStart)
  const active = content.slice(winStart, i)
  const after = content.slice(i)

  let highlightClass = headActive
    ? 'from-[#61dbb4]/45 via-[#4f9dff]/50 to-[#8b7cff]/45 shadow-[0_0_24px_rgba(97,219,180,0.28)] ring-[#61dbb4]/35'
    : 'from-[#61dbb4]/25 via-[#3d7dd6]/35 to-[#6b5fd6]/30 shadow-[0_0_16px_rgba(79,157,255,0.12)] ring-white/10'

  if (hold) {
    highlightClass =
      'from-[#61dbb4]/50 via-[#5aa9ff]/55 to-[#9b8cff]/50 shadow-[0_0_32px_rgba(97,219,180,0.38)] ring-[#61dbb4]/45 animate-pulse'
  }

  return (
    <>
      {before}
      <span
        className={`mx-0.5 inline rounded-lg bg-gradient-to-r px-1.5 py-0.5 align-baseline text-zinc-50 ring-1 ring-inset box-decoration-clone ${highlightClass} ${
          headActive && !hold ? 'transition-all duration-150 ease-out motion-reduce:transition-none' : ''
        }`}
        title={hold ? 'Prefill… waiting for first token' : 'Prefill (visual)'}
      >
        {active}
      </span>
      {after}
    </>
  )
}

export function ChatBox() {
  const [input, setInput] = useState('')
  const messages = useInfStore((s) => s.messages)
  const addUserMessage = useInfStore((s) => s.addUserMessage)
  const isGenerating = useInfStore((s) => s.isGenerating)
  const { sendPrompt, stopInference } = useInferenceSocket()
  const { prefillUserIndex, prefillScanIdx, prefillHeadActive, prefillHold } = usePrefillScan(
    messages,
    isGenerating,
  )

  const onSend = () => {
    if (isGenerating) {
      stopInference()
      return
    }
    const prompt = input.trim()
    if (!prompt) return
    addUserMessage(prompt)
    setInput('')
    sendPrompt(prompt)
  }

  return (
    <section className='flex min-h-screen flex-1 flex-col pt-16 md:pl-64'>
      <div className='mx-auto mb-32 w-full max-w-5xl flex-1 space-y-6 overflow-y-auto px-6 py-10'>
        {messages.map((m, i) => (
          <div key={`${m.role}-${i}`} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
            <div
              className={`w-full text-sm leading-relaxed ${
                m.role === 'user'
                  ? 'max-w-[88%] rounded-2xl border border-white/5 bg-[#1b1f2a] px-6 py-4 text-zinc-100 shadow-[0_0_0_1px_rgba(255,255,255,0.02)]'
                  : 'max-w-[92%] rounded-2xl border border-[#3d4a44]/45 bg-[#1a1e27] px-6 py-5 text-zinc-200 shadow-[0_0_18px_rgba(97,219,180,0.06)]'
              }`}
            >
              {m.role === 'assistant' ? (
                <div className='mb-3 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-[#61dbb4]'>
                  <span className='inline-block h-1.5 w-1.5 rounded-full bg-[#61dbb4]' />
                  LLAMA-3-8B-INSTRUCT
                </div>
              ) : null}
              <div className='whitespace-pre-wrap [overflow-wrap:anywhere]'>
                {m.role === 'user' ? (
                  <UserPromptWithPrefillCaret
                    content={m.content}
                    scanIdx={i === prefillUserIndex ? prefillScanIdx : null}
                    headActive={i === prefillUserIndex ? prefillHeadActive : false}
                    hold={i === prefillUserIndex ? prefillHold : false}
                  />
                ) : (
                  m.content || '...'
                )}
              </div>
              {m.role === 'assistant' ? <span className='ml-1 inline-block h-4 w-1 animate-pulse bg-[#61dbb4] align-middle' /> : null}
              {m.metrics ? <MetricsPanel metrics={m.metrics} /> : null}
            </div>
          </div>
        ))}
      </div>
      <div className='fixed bottom-0 left-0 right-0 z-30 p-4 md:left-64 md:p-6'>
        <div className='mx-auto max-w-5xl'>
          <div className='glass-panel group relative rounded-2xl border border-[#3d4a44]/50 bg-[#181c25]/85 p-2 shadow-[0_0_22px_rgba(97,219,180,0.05)] backdrop-blur'>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                onSend()
              }
            }}
            placeholder='Send via web interface...'
            className='min-h-14 w-full resize-y border-none bg-transparent px-4 py-3 text-sm text-zinc-100 placeholder:text-slate-500 focus:outline-none'
          />
          <div className='flex items-center justify-end px-3 pb-2'>
            <button
              onClick={onSend}
              className='rounded-xl bg-[#61dbb4] px-4 py-2 text-sm font-semibold text-[#00382a] transition hover:bg-[#7ff8cf]'
            >
              {isGenerating ? 'Stop' : 'Send'}
            </button>
          </div>
          <div className='absolute bottom-0 left-1/2 h-[2px] w-0 -translate-x-1/2 bg-[#61dbb4] transition-all duration-500 group-focus-within:w-[97%]' />
          </div>
        </div>
      </div>
    </section>
  )
}
