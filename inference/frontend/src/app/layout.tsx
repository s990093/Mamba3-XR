import './globals.css'
import type { Metadata } from 'next'
import type { ReactNode } from 'react'

export const metadata: Metadata = {
  title: 'Inf-Platform',
  description: 'Inference and monitoring platform',
}

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang='zh-Hant' suppressHydrationWarning>
      <body suppressHydrationWarning>{children}</body>
    </html>
  )
}
