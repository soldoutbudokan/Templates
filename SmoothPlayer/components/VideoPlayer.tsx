'use client'

import { useRef, useState, useCallback, useEffect } from 'react'
import { useVideoPlayer } from '@/lib/useVideoPlayer'
import { useKeyboardShortcuts } from '@/lib/useKeyboardShortcuts'
import VideoControls from './VideoControls'
import KeyboardShortcutsHelp from './KeyboardShortcutsHelp'

interface VideoPlayerProps {
  url: string
  onChangeUrl: () => void
}

export default function VideoPlayer({ url, onChangeUrl }: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const hideTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const { state, controls, setIsSeeking } = useVideoPlayer(videoRef, containerRef)
  const [controlsVisible, setControlsVisible] = useState(true)
  const [showHelp, setShowHelp] = useState(false)
  const [cursorHidden, setCursorHidden] = useState(false)

  const showControls = useCallback(() => {
    setControlsVisible(true)
    setCursorHidden(false)
    if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    hideTimerRef.current = setTimeout(() => {
      if (!showHelp) {
        setControlsVisible(false)
        setCursorHidden(true)
      }
    }, 3000)
  }, [showHelp])

  // Show controls on mount, start hide timer
  useEffect(() => {
    showControls()
    return () => {
      if (hideTimerRef.current) clearTimeout(hideTimerRef.current)
    }
  }, [showControls])

  useKeyboardShortcuts(state, controls, showControls, setShowHelp)

  const onVideoClick = useCallback(() => {
    controls.togglePlay()
    showControls()
  }, [controls, showControls])

  const onDoubleClick = useCallback(() => {
    controls.toggleFullscreen()
  }, [controls])

  // Separate click and double-click handling
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const handleClick = useCallback(() => {
    if (clickTimerRef.current) {
      clearTimeout(clickTimerRef.current)
      clickTimerRef.current = null
      onDoubleClick()
    } else {
      clickTimerRef.current = setTimeout(() => {
        clickTimerRef.current = null
        onVideoClick()
      }, 200)
    }
  }, [onVideoClick, onDoubleClick])

  return (
    <div
      ref={containerRef}
      className={`relative w-full h-screen bg-black flex items-center justify-center ${cursorHidden ? 'cursor-none' : ''}`}
      onMouseMove={showControls}
    >
      <video
        ref={videoRef}
        src={url}
        style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain' }}
        preload="auto"
        onClick={handleClick}
      />

      {/* Loading spinner */}
      {state.isLoading && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-12 h-12 border-4 border-white/20 border-t-white rounded-full animate-spin" />
        </div>
      )}

      {/* Error display */}
      {state.error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/60">
          <div className="text-center max-w-sm mx-4">
            <p className="text-red-400 mb-4">{state.error}</p>
            <button
              onClick={onChangeUrl}
              className="bg-white/10 hover:bg-white/20 text-white px-4 py-2 rounded-lg transition-colors text-sm"
            >
              Try a different URL
            </button>
          </div>
        </div>
      )}

      {/* Change URL button */}
      <button
        onClick={onChangeUrl}
        className={`absolute top-4 left-4 text-white/40 hover:text-white text-xs bg-black/30 hover:bg-black/50 px-3 py-1.5 rounded-lg transition-all duration-300 ${
          controlsVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'
        }`}
      >
        Change URL
      </button>

      {/* Controls overlay */}
      <VideoControls
        state={state}
        controls={controls}
        visible={controlsVisible}
        onSeekStart={() => setIsSeeking(true)}
        onSeekEnd={() => setIsSeeking(false)}
        onShowHelp={() => setShowHelp(true)}
      />

      {/* Keyboard shortcuts help */}
      {showHelp && <KeyboardShortcutsHelp onClose={() => setShowHelp(false)} />}
    </div>
  )
}
