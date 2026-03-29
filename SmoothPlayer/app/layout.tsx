import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'SmoothPlayer',
  description: 'Paste an MP4 link and play with smooth controls and keyboard shortcuts',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-neutral-950 text-white">
        {children}
      </body>
    </html>
  )
}
