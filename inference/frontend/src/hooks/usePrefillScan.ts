'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import type { ChatMessage } from '@/store/useInfStore'

const HEAD = 10

function easeInQuad(t: number): number {
  return t * t
}

/**
 * Progress 0..1 with slow start so early characters (incl. first line) stay on screen longer.
 */
function easeInCubic(t: number): number {
  return t * t * t
}

/**
 * Map elapsed time -> caret index. Uses ease-in so the sweep does not "finish the first line"
 * in a fraction of a second; long prompts get extra budget for the first line.
 */
function indexFromElapsed(elapsedMs: number, len: number, totalMs: number, firstLineLen: number): number {
  if (len <= 0) return 0
  const t = Math.min(1, elapsedMs / totalMs)
  // Blend: mostly ease-in on full length, with extra weight on first line in time domain
  const lineWeight = Math.min(1, firstLineLen / Math.max(1, len))
  const p = (1 - lineWeight) * easeInCubic(t) + lineWeight * easeInQuad(t)
  return Math.min(len, Math.floor(p * len))
}

/**
 * Prefill UX: gradient highlight sweeps the last user prompt while assistant is still empty.
 * Longer min duration + slow-start curve; after sweep completes, UI can show "hold" state.
 */
export function usePrefillScan(messages: ChatMessage[], isGenerating: boolean) {
  const [scanIdx, setScanIdx] = useState(0)
  const messagesRef = useRef(messages)
  messagesRef.current = messages

  const prefill = useMemo(() => {
    const n = messages.length
    if (n < 2 || !isGenerating) return null
    const assistant = messages[n - 1]
    const user = messages[n - 2]
    if (assistant.role !== 'assistant' || user.role !== 'user') return null
    if (assistant.content.length > 0) return null
    const text = user.content
    const len = text.length
    if (len === 0) return null
    const firstLineLen = text.indexOf('\n') === -1 ? len : text.indexOf('\n')
    return { userIndex: n - 2, len, firstLineLen }
  }, [messages, isGenerating])

  useEffect(() => {
    if (!prefill) {
      setScanIdx(0)
      return
    }

    const { len, firstLineLen } = prefill
    // Longer floor so short prompts / first line do not "blink" past; scale with length + first line
    const totalMs = Math.min(
      22_000,
      Math.max(4200, len * 24 + Math.min(firstLineLen, 400) * 28),
    )
    const start = performance.now()
    let raf = 0
    let lastEmitted = -1

    const step = (now: number) => {
      const msgs = messagesRef.current
      const last = msgs[msgs.length - 1]
      if (!last || last.role !== 'assistant' || last.content.length > 0) {
        setScanIdx(0)
        return
      }

      const elapsed = now - start
      const raw = elapsed >= totalMs ? len : indexFromElapsed(elapsed, len, totalMs, firstLineLen)

      if (raw !== lastEmitted) {
        lastEmitted = raw
        setScanIdx(raw)
      }

      if (elapsed < totalMs) {
        raf = requestAnimationFrame(step)
      }
    }

    raf = requestAnimationFrame(step)
    return () => cancelAnimationFrame(raf)
  }, [prefill?.userIndex, prefill?.len, prefill?.firstLineLen, isGenerating])

  const atEnd = Boolean(prefill && scanIdx >= prefill.len)
  const prefillHold = Boolean(prefill && atEnd)

  return {
    prefillUserIndex: prefill?.userIndex ?? null,
    prefillScanIdx: prefill ? scanIdx : null,
    /** True while caret is in the first ~10 chars (stronger gradient). */
    prefillHeadActive: prefill ? scanIdx < Math.min(HEAD, prefill.len) : false,
    /** Sweep finished but assistant still empty — show "waiting on prefill" styling. */
    prefillHold,
  }
}
