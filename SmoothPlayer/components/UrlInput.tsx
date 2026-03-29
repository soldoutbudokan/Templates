'use client'

import { useState, FormEvent } from 'react'

interface UrlInputProps {
  onSubmit: (url: string) => void
}

export default function UrlInput({ onSubmit }: UrlInputProps) {
  const [url, setUrl] = useState('')
  const [error, setError] = useState('')

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault()
    const trimmed = url.trim()
    if (!trimmed) {
      setError('Please enter a URL')
      return
    }
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      setError('URL must start with http:// or https://')
      return
    }
    setError('')
    onSubmit(trimmed)
  }

  return (
    <div className="flex items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-lg">
        <h1 className="text-3xl font-bold text-white mb-2 text-center">SmoothPlayer</h1>
        <p className="text-white/50 text-sm text-center mb-8">Paste an MP4 link and play with smooth controls</p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <input
              type="url"
              value={url}
              onChange={e => { setUrl(e.target.value); setError('') }}
              placeholder="https://example.com/video.mp4"
              autoFocus
              className="w-full bg-white/5 border border-white/10 rounded-lg px-4 py-3 text-white placeholder-white/30 focus:outline-none focus:border-player-accent focus:ring-1 focus:ring-player-accent transition-colors text-sm"
            />
            {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
          </div>

          <button
            type="submit"
            className="w-full bg-player-accent hover:bg-blue-600 text-white font-medium py-3 rounded-lg transition-colors"
          >
            Play
          </button>
        </form>

        <div className="mt-8 text-center text-white/30 text-xs">
          Press <kbd className="bg-white/10 px-1.5 py-0.5 rounded font-mono">?</kbd> during playback for keyboard shortcuts
        </div>
      </div>
    </div>
  )
}
