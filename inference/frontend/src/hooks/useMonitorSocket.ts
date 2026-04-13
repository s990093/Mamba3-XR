'use client'

import { useEffect } from 'react'

import { useInfStore } from '@/store/useInfStore'
import type { MonitorPayload } from '@/types/metrics'

const wsBase = process.env.NEXT_PUBLIC_WS_BASE ?? 'ws://localhost:8000'

/**
 * Monitor updates come only from WebSocket (same payload as GET /api/status).
 * We intentionally avoid fetch() on mount: some browser extensions wrap `window.fetch`
 * and throw in ways that still surface in the Next.js error overlay.
 */
export function useMonitorSocket(): void {
  const setMonitor = useInfStore((s) => s.setMonitor)

  useEffect(() => {
    let ws: WebSocket | null = null
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null
    let retryCount = 0
    let stopped = false

    const connect = () => {
      if (stopped) return
      ws = new WebSocket(`${wsBase}/ws/monitor`)
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as { type?: string; value?: MonitorPayload }
          if (data.type === 'monitor' && data.value) {
            setMonitor(data.value)
          }
        } catch {
          // Ignore malformed messages during reconnects.
        }
      }
      ws.onopen = () => {
        retryCount = 0
      }
      ws.onclose = () => {
        if (stopped) return
        const delay = Math.min(1000 * 2 ** retryCount, 10000)
        retryCount += 1
        reconnectTimer = setTimeout(connect, delay)
      }
    }
    connect()

    return () => {
      stopped = true
      if (reconnectTimer) clearTimeout(reconnectTimer)
      ws?.close()
    }
  }, [setMonitor])
}
