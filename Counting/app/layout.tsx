import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Card Counting Trainer',
  description: 'Build a steady Hi-Lo count with guided shoe practice, checkpoint feedback, replay, and saved progress.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-felt-green text-white">
        {children}
      </body>
    </html>
  )
}
