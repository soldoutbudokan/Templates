'use client'

import { useState, useCallback, useEffect, useRef } from 'react'
import { PlayerState, PlayerControls } from './types'

export function useVideoPlayer(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  containerRef: React.RefObject<HTMLDivElement | null>
): { state: PlayerState; controls: PlayerControls; isSeeking: boolean; setIsSeeking: (v: boolean) => void } {
  const [state, setState] = useState<PlayerState>({
    isPlaying: false,
    currentTime: 0,
    duration: 0,
    volume: 1,
    isMuted: false,
    isFullscreen: false,
    buffered: [],
    isLoading: false,
    error: null,
    playbackRate: 1,
  })

  const [isSeeking, setIsSeeking] = useState(false)
  const rafRef = useRef<number | null>(null)

  // Restore volume from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('smoothplayer-volume')
    if (saved !== null) {
      const vol = parseFloat(saved)
      if (isFinite(vol) && vol >= 0 && vol <= 1) {
        setState(s => ({ ...s, volume: vol }))
        if (videoRef.current) videoRef.current.volume = vol
      }
    }
  }, [videoRef])

  // RAF loop for smooth seek bar updates
  useEffect(() => {
    const tick = () => {
      if (videoRef.current && !isSeeking) {
        setState(s => ({ ...s, currentTime: videoRef.current!.currentTime }))
      }
      rafRef.current = requestAnimationFrame(tick)
    }
    rafRef.current = requestAnimationFrame(tick)
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current)
    }
  }, [videoRef, isSeeking])

  // Video event listeners
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const onDurationChange = () => setState(s => ({ ...s, duration: video.duration }))
    const onPlay = () => setState(s => ({ ...s, isPlaying: true }))
    const onPause = () => setState(s => ({ ...s, isPlaying: false }))
    const onEnded = () => setState(s => ({ ...s, isPlaying: false }))
    const onWaiting = () => setState(s => ({ ...s, isLoading: true }))
    const onCanPlay = () => setState(s => ({ ...s, isLoading: false }))
    const onVolumeChange = () => {
      setState(s => ({ ...s, volume: video.volume, isMuted: video.muted }))
    }
    const onProgress = () => {
      const ranges: { start: number; end: number }[] = []
      for (let i = 0; i < video.buffered.length; i++) {
        ranges.push({ start: video.buffered.start(i), end: video.buffered.end(i) })
      }
      setState(s => ({ ...s, buffered: ranges }))
    }
    const onError = () => {
      const code = video.error?.code
      let msg = 'Something went wrong. Please try a different URL.'
      if (code === 2) msg = 'Network error. Check your connection and URL.'
      if (code === 3) msg = 'Video decoding failed. The file may be corrupted.'
      if (code === 4) msg = 'Unsupported video format or blocked by CORS. Try a direct download link.'
      setState(s => ({ ...s, error: msg, isLoading: false }))
    }

    video.addEventListener('durationchange', onDurationChange)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('ended', onEnded)
    video.addEventListener('waiting', onWaiting)
    video.addEventListener('canplay', onCanPlay)
    video.addEventListener('volumechange', onVolumeChange)
    video.addEventListener('progress', onProgress)
    video.addEventListener('error', onError)

    return () => {
      video.removeEventListener('durationchange', onDurationChange)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('ended', onEnded)
      video.removeEventListener('waiting', onWaiting)
      video.removeEventListener('canplay', onCanPlay)
      video.removeEventListener('volumechange', onVolumeChange)
      video.removeEventListener('progress', onProgress)
      video.removeEventListener('error', onError)
    }
  }, [videoRef])

  // Fullscreen change listener
  useEffect(() => {
    const onFsChange = () => {
      setState(s => ({ ...s, isFullscreen: !!document.fullscreenElement }))
    }
    document.addEventListener('fullscreenchange', onFsChange)
    return () => document.removeEventListener('fullscreenchange', onFsChange)
  }, [])

  const togglePlay = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) video.play()
    else video.pause()
  }, [videoRef])

  const seek = useCallback((time: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.max(0, Math.min(time, video.duration || 0))
  }, [videoRef])

  const skip = useCallback((seconds: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.max(0, Math.min(video.currentTime + seconds, video.duration || 0))
  }, [videoRef])

  const setVolume = useCallback((vol: number) => {
    const video = videoRef.current
    if (!video) return
    const clamped = Math.max(0, Math.min(1, vol))
    video.volume = clamped
    video.muted = false
    localStorage.setItem('smoothplayer-volume', String(clamped))
  }, [videoRef])

  const toggleMute = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    video.muted = !video.muted
  }, [videoRef])

  const toggleFullscreen = useCallback(() => {
    const container = containerRef.current
    if (!container) return
    if (document.fullscreenElement) {
      document.exitFullscreen()
    } else {
      if (container.requestFullscreen) {
        container.requestFullscreen()
      } else if ((container as any).webkitRequestFullscreen) {
        (container as any).webkitRequestFullscreen()
      }
    }
  }, [containerRef])

  const jumpToPercent = useCallback((percent: number) => {
    const video = videoRef.current
    if (!video || !video.duration) return
    video.currentTime = (percent / 100) * video.duration
  }, [videoRef])

  const setPlaybackRate = useCallback((rate: number) => {
    const video = videoRef.current
    if (!video) return
    const clamped = Math.max(0.25, Math.min(4, rate))
    video.playbackRate = clamped
    setState(s => ({ ...s, playbackRate: clamped }))
  }, [videoRef])

  return {
    state,
    controls: {
      togglePlay,
      seek,
      skip,
      setVolume,
      toggleMute,
      toggleFullscreen,
      jumpToPercent,
      setPlaybackRate,
    },
    isSeeking,
    setIsSeeking,
  }
}
