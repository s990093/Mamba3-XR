'use client'

import { useEffect, useState } from 'react'

import { ChatBox } from '@/components/ChatBox'
import { SettingsModal } from '@/components/SettingsModal'
import { Sidebar } from '@/components/Sidebar'
import { TopMonitorBar } from '@/components/TopMonitorBar'
import { useMonitorSocket } from '@/hooks/useMonitorSocket'

export default function HomePage() {
  const [mounted, setMounted] = useState(false)
  const [openSettings, setOpenSettings] = useState(false)
  useMonitorSocket()
  useEffect(() => setMounted(true), [])

  if (!mounted) {
    return <main className='min-h-screen bg-zinc-950 text-zinc-100' />
  }

  return (
    <main className='min-h-screen bg-[#11131d] text-zinc-100'>
      <TopMonitorBar onOpenSettings={() => setOpenSettings(true)} />
      <Sidebar />
      <ChatBox />
      <SettingsModal open={openSettings} onClose={() => setOpenSettings(false)} />
    </main>
  )
}
