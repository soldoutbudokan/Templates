'use client'

import { PlayerState, PlayerControls } from '@/lib/types'
import { formatTime } from '@/lib/formatTime'
import SeekBar from './SeekBar'
import VolumeControl from './VolumeControl'

interface VideoControlsProps {
  state: PlayerState
  controls: PlayerControls
  visible: boolean
  onSeekStart: () => void
  onSeekEnd: () => void
  onShowHelp: () => void
}

export default function VideoControls({ state, controls, visible, onSeekStart, onSeekEnd, onShowHelp }: VideoControlsProps) {
  return (
    <div
      className={`absolute bottom-0 left-0 right-0 transition-opacity duration-300 ${
        visible ? 'opacity-100' : 'opacity-0 pointer-events-none'
      }`}
    >
      {/* Gradient background */}
      <div className="bg-gradient-to-t from-black/80 via-black/40 to-transparent pt-16 pb-3 px-4">
        {/* Seek bar */}
        <SeekBar
          currentTime={state.currentTime}
          duration={state.duration}
          buffered={state.buffered}
          onSeek={controls.seek}
          onSeekStart={onSeekStart}
          onSeekEnd={onSeekEnd}
        />

        {/* Controls row */}
        <div className="flex items-center gap-2 mt-1">
          {/* Play/Pause */}
          <button
            onClick={controls.togglePlay}
            className="text-white/90 hover:text-white transition-colors p-1"
            title={state.isPlaying ? 'Pause (Space)' : 'Play (Space)'}
          >
            {state.isPlaying ? (
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z"/>
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-6 h-6">
                <path d="M8 5v14l11-7z"/>
              </svg>
            )}
          </button>

          {/* Time */}
          <span className="text-white/80 text-xs font-mono tabular-nums select-none">
            {formatTime(state.currentTime)} / {formatTime(state.duration)}
          </span>

          {/* Spacer */}
          <div className="flex-1" />

          {/* Playback rate (only show if not 1x) */}
          {state.playbackRate !== 1 && (
            <span className="text-white/60 text-xs font-mono select-none">
              {state.playbackRate}x
            </span>
          )}

          {/* Volume */}
          <VolumeControl
            volume={state.volume}
            isMuted={state.isMuted}
            onVolumeChange={controls.setVolume}
            onToggleMute={controls.toggleMute}
          />

          {/* Help */}
          <button
            onClick={onShowHelp}
            className="text-white/50 hover:text-white transition-colors p-1 text-sm font-mono"
            title="Keyboard shortcuts (?)"
          >
            ?
          </button>

          {/* Fullscreen */}
          <button
            onClick={controls.toggleFullscreen}
            className="text-white/80 hover:text-white transition-colors p-1"
            title={state.isFullscreen ? 'Exit fullscreen (F)' : 'Fullscreen (F)'}
          >
            {state.isFullscreen ? (
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M5 16h3v3h2v-5H5v2zm3-8H5v2h5V5H8v3zm6 11h2v-3h3v-2h-5v5zm2-11V5h-2v5h5V8h-3z"/>
              </svg>
            ) : (
              <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
                <path d="M7 14H5v5h5v-2H7v-3zm-2-4h2V7h3V5H5v5zm12 7h-3v2h5v-5h-2v3zM14 5v2h3v3h2V5h-5z"/>
              </svg>
            )}
          </button>
        </div>
      </div>
    </div>
  )
}
