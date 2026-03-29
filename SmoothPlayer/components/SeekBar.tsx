'use client'

import { useRef, useState, useCallback } from 'react'
import { formatTime } from '@/lib/formatTime'

interface SeekBarProps {
  currentTime: number
  duration: number
  buffered: { start: number; end: number }[]
  onSeek: (time: number) => void
  onSeekStart: () => void
  onSeekEnd: () => void
}

export default function SeekBar({ currentTime, duration, buffered, onSeek, onSeekStart, onSeekEnd }: SeekBarProps) {
  const trackRef = useRef<HTMLDivElement>(null)
  const [hoverTime, setHoverTime] = useState<number | null>(null)
  const [hoverX, setHoverX] = useState(0)
  const [isDragging, setIsDragging] = useState(false)

  const getTimeFromX = useCallback((clientX: number) => {
    const track = trackRef.current
    if (!track || !duration) return 0
    const rect = track.getBoundingClientRect()
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    return ratio * duration
  }, [duration])

  const onMouseDown = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(true)
    onSeekStart()
    const time = getTimeFromX(e.clientX)
    onSeek(time)

    const onMouseMove = (e: MouseEvent) => {
      const time = getTimeFromX(e.clientX)
      onSeek(time)
    }

    const onMouseUp = () => {
      setIsDragging(false)
      onSeekEnd()
      window.removeEventListener('mousemove', onMouseMove)
      window.removeEventListener('mouseup', onMouseUp)
    }

    window.addEventListener('mousemove', onMouseMove)
    window.addEventListener('mouseup', onMouseUp)
  }, [getTimeFromX, onSeek, onSeekStart, onSeekEnd])

  const onTouchStart = useCallback((e: React.TouchEvent) => {
    setIsDragging(true)
    onSeekStart()
    const time = getTimeFromX(e.touches[0].clientX)
    onSeek(time)

    const onTouchMove = (e: TouchEvent) => {
      const time = getTimeFromX(e.touches[0].clientX)
      onSeek(time)
    }

    const onTouchEnd = () => {
      setIsDragging(false)
      onSeekEnd()
      window.removeEventListener('touchmove', onTouchMove)
      window.removeEventListener('touchend', onTouchEnd)
    }

    window.addEventListener('touchmove', onTouchMove)
    window.addEventListener('touchend', onTouchEnd)
  }, [getTimeFromX, onSeek, onSeekStart, onSeekEnd])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const time = getTimeFromX(e.clientX)
    setHoverTime(time)
    const track = trackRef.current
    if (track) {
      const rect = track.getBoundingClientRect()
      setHoverX(e.clientX - rect.left)
    }
  }, [getTimeFromX])

  const progress = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div className="group/seek flex-1 flex items-center px-2">
      <div
        ref={trackRef}
        className="relative w-full h-1 group-hover/seek:h-1.5 transition-all duration-150 cursor-pointer"
        onMouseDown={onMouseDown}
        onTouchStart={onTouchStart}
        onMouseMove={onMouseMove}
        onMouseLeave={() => setHoverTime(null)}
      >
        {/* Background track */}
        <div className="absolute inset-0 bg-white/20 rounded-full" />

        {/* Buffered ranges */}
        {buffered.map((range, i) => {
          const left = duration > 0 ? (range.start / duration) * 100 : 0
          const width = duration > 0 ? ((range.end - range.start) / duration) * 100 : 0
          return (
            <div
              key={i}
              className="absolute top-0 bottom-0 bg-white/30 rounded-full"
              style={{ left: `${left}%`, width: `${width}%` }}
            />
          )
        })}

        {/* Played progress */}
        <div
          className="absolute top-0 bottom-0 left-0 bg-player-accent rounded-full"
          style={{ width: `${progress}%` }}
        />

        {/* Thumb */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 bg-player-accent rounded-full shadow transition-opacity duration-150 ${
            isDragging ? 'opacity-100 scale-125' : 'opacity-0 group-hover/seek:opacity-100'
          }`}
          style={{ left: `${progress}%` }}
        />

        {/* Hover time tooltip */}
        {hoverTime !== null && (
          <div
            className="absolute -top-8 -translate-x-1/2 bg-black/90 text-white text-xs px-2 py-1 rounded pointer-events-none"
            style={{ left: `${hoverX}px` }}
          >
            {formatTime(hoverTime)}
          </div>
        )}
      </div>
    </div>
  )
}
