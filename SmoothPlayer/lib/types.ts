export interface PlayerState {
  isPlaying: boolean
  currentTime: number
  duration: number
  volume: number
  isMuted: boolean
  isFullscreen: boolean
  buffered: { start: number; end: number }[]
  isLoading: boolean
  error: string | null
  playbackRate: number
}

export interface PlayerControls {
  togglePlay: () => void
  seek: (time: number) => void
  skip: (seconds: number) => void
  setVolume: (vol: number) => void
  toggleMute: () => void
  toggleFullscreen: () => void
  jumpToPercent: (percent: number) => void
  setPlaybackRate: (rate: number) => void
}
