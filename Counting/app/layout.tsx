import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Card Counting Trainer',
  description: 'Practice Hi-Lo card counting with realistic deck simulation and speed drills',
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
