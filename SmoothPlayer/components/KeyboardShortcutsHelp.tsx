'use client'

interface KeyboardShortcutsHelpProps {
  onClose: () => void
}

const shortcuts = [
  { keys: 'Space / K', action: 'Play / Pause' },
  { keys: '← →', action: 'Skip 2s' },
  { keys: 'Shift + ← →', action: 'Skip 10s' },
  { keys: '↑ ↓', action: 'Volume' },
  { keys: 'M', action: 'Mute' },
  { keys: 'F', action: 'Fullscreen' },
  { keys: '0-9', action: 'Jump to 0%-90%' },
  { keys: '< >', action: 'Playback speed' },
  { keys: '?', action: 'This help' },
  { keys: 'Esc', action: 'Exit fullscreen / Close' },
]

export default function KeyboardShortcutsHelp({ onClose }: KeyboardShortcutsHelpProps) {
  return (
    <div
      className="absolute inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="bg-neutral-900 border border-white/10 rounded-xl p-6 max-w-sm w-full mx-4 shadow-2xl"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">Keyboard Shortcuts</h2>
          <button onClick={onClose} className="text-white/50 hover:text-white transition-colors">
            <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/>
            </svg>
          </button>
        </div>
        <div className="space-y-2">
          {shortcuts.map(({ keys, action }) => (
            <div key={keys} className="flex items-center justify-between text-sm">
              <span className="text-white/60">{action}</span>
              <kbd className="bg-white/10 text-white/90 px-2 py-0.5 rounded text-xs font-mono">{keys}</kbd>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
