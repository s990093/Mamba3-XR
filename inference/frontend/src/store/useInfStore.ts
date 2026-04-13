import { create } from 'zustand'

import type { MetricsPayload, MonitorPayload } from '@/types/metrics'

export type ChatMessage = {
  role: 'user' | 'assistant'
  content: string
  metrics?: MetricsPayload
}

export type InferenceSettings = {
  top_k: number
  top_p: number
  min_p: number
  rep_pen: number
  pres_pen: number
  freq_pen: number
  temperature: number
  max_tokens: number
  no_eos_stop: boolean
  system_prompt: string
}

type InfState = {
  messages: ChatMessage[]
  isGenerating: boolean
  monitor?: MonitorPayload
  settings: InferenceSettings
  setSettings: (s: InferenceSettings) => void
  addUserMessage: (content: string) => void
  beginAssistantMessage: () => void
  appendAssistantToken: (token: string) => void
  attachMetricsToLastMessage: (metrics: MetricsPayload) => void
  setGenerating: (value: boolean) => void
  setMonitor: (value: MonitorPayload) => void
  resetChat: () => void
}

const defaultSettings: InferenceSettings = {
  top_k: 0,
  top_p: 0.9,
  min_p: 0.0,
  rep_pen: 1.05,
  pres_pen: 0.0,
  freq_pen: 0.05,
  temperature: 0.8,
  max_tokens: 256,
  no_eos_stop: false,
  system_prompt: 'You are a helpful assistant.',
}

export const useInfStore = create<InfState>((set) => ({
  messages: [],
  isGenerating: false,
  settings: defaultSettings,
  setSettings: (settings) => set({ settings }),
  addUserMessage: (content) =>
    set((state) => ({ messages: [...state.messages, { role: 'user', content }] })),
  beginAssistantMessage: () =>
    set((state) => ({ messages: [...state.messages, { role: 'assistant', content: '' }] })),
  appendAssistantToken: (token) =>
    set((state) => {
      const messages = [...state.messages]
      const idx = messages.length - 1
      if (idx >= 0 && messages[idx].role === 'assistant') {
        messages[idx] = { ...messages[idx], content: messages[idx].content + token }
      }
      return { messages }
    }),
  attachMetricsToLastMessage: (metrics) =>
    set((state) => {
      const messages = [...state.messages]
      const idx = messages.length - 1
      if (idx >= 0 && messages[idx].role === 'assistant') {
        messages[idx] = { ...messages[idx], metrics }
      }
      return { messages }
    }),
  setGenerating: (isGenerating) => set({ isGenerating }),
  setMonitor: (monitor) => set({ monitor }),
  resetChat: () => set({ messages: [] }),
}))
