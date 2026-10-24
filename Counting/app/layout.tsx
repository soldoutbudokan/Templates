import type { Metadata } from 'next'
import './globals.css'
import './game.css'
import './sync.css'
import './challenge.css'

export const metadata: Metadata = {
  title: 'Card Counting Trainer',
  description: 'Play a six-hand blackjack challenge. Keep the count, choose practice bets, and build a steadier game with precise feedback.',
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
