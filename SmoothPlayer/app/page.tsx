'use client'

import { useState } from 'react'
import UrlInput from '@/components/UrlInput'
import VideoPlayer from '@/components/VideoPlayer'

export default function Home() {
  const [videoUrl, setVideoUrl] = useState<string | null>(null)

  if (videoUrl) {
    return <VideoPlayer url={videoUrl} onChangeUrl={() => setVideoUrl(null)} />
  }

  return <UrlInput onSubmit={setVideoUrl} />
}
