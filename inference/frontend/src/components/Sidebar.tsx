'use client'

import { useInfStore } from '@/store/useInfStore'

const apiBase = process.env.NEXT_PUBLIC_API_BASE ?? 'http://localhost:8000'

export function Sidebar() {
  const resetChat = useInfStore((s) => s.resetChat)

  const onNewChat = async () => {
    await fetch(`${apiBase}/api/rest`, { method: 'POST' })
    resetChat()
  }

  return (
    <aside className='fixed left-0 top-0 hidden h-screen w-64 flex-col border-r border-[#3d4a44]/30 bg-[#1a1b26] px-3 pb-4 pt-20 md:flex'>
      <div className='mb-6 px-3'>
        <div className='text-lg font-bold text-white'>The Observatory</div>
        <div className='text-xs text-slate-500'>v2.4.0-alpha</div>
      </div>
      <div className='mb-5 px-1'>
        <button
          onClick={onNewChat}
          className='w-full rounded-lg bg-gradient-to-r from-[#61dbb4] to-[#12a480] px-3 py-2 text-sm font-medium text-[#00382a] shadow-lg shadow-[#61dbb4]/10'
        >
          + New Chat
        </button>
      </div>
      <nav className='flex-1 space-y-1 px-1 text-sm'>
        <div className='rounded-md border-l-4 border-[#61dbb4] bg-[#61dbb4]/10 px-3 py-2 text-[#61dbb4]'>Current Session</div>
        <div className='rounded-md px-3 py-2 text-slate-400 hover:bg-[#1e1f2a] hover:text-white'>Conversation History</div>
        <div className='rounded-md px-3 py-2 text-slate-400 hover:bg-[#1e1f2a] hover:text-white'>Model Library</div>
        <div className='rounded-md px-3 py-2 text-slate-400 hover:bg-[#1e1f2a] hover:text-white'>System Logs</div>
      </nav>
      <div className='mt-6 border-t border-[#3d4a44]/30 px-2 pt-3 text-xs text-slate-500'>
        <div className='py-1'>Docs</div>
        <div className='py-1'>Support</div>
      </div>
    </aside>
  )
}
