'use client'

import { useEffect } from 'react'
import { PlayerState, PlayerControls } from './types'

export function useKeyboardShortcuts(
  state: PlayerState,
  controls: PlayerControls,
  showControls: () => void,
  setShowHelp: (v: boolean | ((prev: boolean) => boolean)) => void
) {
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Don't capture keys when typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return

      const key = e.key
      const shift = e.shiftKey

      switch (key) {
        case ' ':
        case 'k':
        case 'K':
          e.preventDefault()
          controls.togglePlay()
          showControls()
          break
        case 'ArrowLeft':
          e.preventDefault()
          controls.skip(shift ? -10 : -2)
          showControls()
          break
        case 'ArrowRight':
          e.preventDefault()
          controls.skip(shift ? 10 : 2)
          showControls()
          break
        case 'ArrowUp':
          e.preventDefault()
          controls.setVolume(state.volume + 0.05)
          showControls()
          break
        case 'ArrowDown':
          e.preventDefault()
          controls.setVolume(state.volume - 0.05)
          showControls()
          break
        case 'm':
        case 'M':
          controls.toggleMute()
          showControls()
          break
        case 'f':
        case 'F':
          controls.toggleFullscreen()
          break
        case '>':
          controls.setPlaybackRate(state.playbackRate + 0.25)
          showControls()
          break
        case '<':
          controls.setPlaybackRate(state.playbackRate - 0.25)
          showControls()
          break
        case '?':
          setShowHelp((prev: boolean) => !prev)
          break
        case 'Escape':
          if (state.isFullscreen) {
            controls.toggleFullscreen()
          } else {
            setShowHelp(false)
          }
          break
        default:
          // Number keys 0-9 for percentage jump
          if (key >= '0' && key <= '9' && !shift) {
            const percent = parseInt(key) * 10
            controls.jumpToPercent(percent)
            showControls()
          }
          break
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [state.volume, state.playbackRate, state.isFullscreen, controls, showControls, setShowHelp])
}
