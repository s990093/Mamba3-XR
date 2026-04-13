'use client'

import { useRef } from 'react'

import { useInfStore } from '@/store/useInfStore'
import type { MetricsPayload } from '@/types/metrics'

const wsBase = process.env.NEXT_PUBLIC_WS_BASE ?? 'ws://localhost:8000'

export function useInferenceSocket() {
  const socketRef = useRef<WebSocket | null>(null)
  const beginAssistantMessage = useInfStore((s) => s.beginAssistantMessage)
  const appendAssistantToken = useInfStore((s) => s.appendAssistantToken)
  const attachMetrics = useInfStore((s) => s.attachMetricsToLastMessage)
  const setGenerating = useInfStore((s) => s.setGenerating)

  const ensureSocket = () => {
    if (socketRef.current && socketRef.current.readyState <= 1) {
      return socketRef.current
    }

    const ws = new WebSocket(`${wsBase}/ws/inf`)
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as { type: string; value?: unknown }
      if (data.type === 'token' && typeof data.value === 'string') {
        appendAssistantToken(data.value)
      }
      if (data.type === 'metrics' && data.value) {
        attachMetrics(data.value as MetricsPayload)
      }
      if (data.type === 'settings_applied' && data.value) {
        // Visibility: confirm backend actually uses the latest settings for this run.
        console.info('[ws] settings_applied:', data.value)
      }
      if (data.type === 'done' || data.type === 'error') {
        setGenerating(false)
      }
    }
    ws.onclose = () => setGenerating(false)

    socketRef.current = ws
    return ws
  }

  const sendPrompt = (prompt: string) => {
    const ws = ensureSocket()
    const doSend = () => {
      beginAssistantMessage()
      setGenerating(true)
      ws.send(JSON.stringify({ type: 'prompt', prompt }))
    }

    if (ws.readyState === WebSocket.OPEN) {
      doSend()
    } else {
      ws.addEventListener('open', doSend, { once: true })
    }
  }

  const stopInference = () => {
    const ws = socketRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: 'stop' }))
    setGenerating(false)
  }

  return { sendPrompt, stopInference }
}
