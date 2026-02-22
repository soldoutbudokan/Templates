# Live Translator

**Live:** [frenchwhisper.vercel.app](https://frenchwhisper.vercel.app)

A simple web app that provides real-time speech-to-text translation to English. Speak in any supported language and see English subtitles appear live.

## Features

- Real-time speech recognition using the Web Speech API
- Translation to English via MyMemory API (free, no API key required)
- 12 supported input languages: French, Spanish, German, Italian, Portuguese, Dutch, Russian, Japanese, Korean, Chinese (Mandarin), Arabic, Hindi
- Works on Chrome (desktop and Android), Safari (macOS), and Edge
- Single HTML file - no build step required

## Usage

1. Open `web/index.html` in a browser (Chrome recommended)
2. Select the language you'll be speaking
3. Tap the microphone button
4. Allow microphone access when prompted
5. Speak - see your words transcribed and translated to English in real-time

## Deployment to Vercel

### Option 1: Vercel CLI

```bash
cd web
npx vercel
```

Follow the prompts to log in and deploy. You'll receive a URL like `https://your-project.vercel.app`.

For production deployment:
```bash
npx vercel --prod
```

### Option 2: GitHub Integration

1. Push this repo to GitHub
2. Go to [vercel.com](https://vercel.com) and sign in
3. Click "Import Project" and select your repo
4. Vercel auto-detects it as a static site
5. Click Deploy

Future pushes to `main` will auto-deploy.

## Browser Compatibility

| Browser | Desktop | Mobile |
|---------|---------|--------|
| Chrome | Yes | Yes (Android) |
| Safari | Yes | Limited (iOS) |
| Edge | Yes | Yes |
| Firefox | No | No |

**Note:** iOS Safari has limited Web Speech API support. For best results on iPhone, use Chrome on Android or desktop browsers.

## How It Works

1. **Speech Recognition**: Uses the browser's built-in `SpeechRecognition` API to capture and transcribe speech
2. **Translation**: Sends transcribed text to MyMemory's free translation API
3. **Display**: Shows both original text and English translation in real-time

## Limitations

- Requires internet connection (speech recognition uses cloud services)
- MyMemory API has rate limits (~1000 words/day for anonymous users)
- Web Speech API requires specifying input language (no auto-detection)
- iOS Safari support is limited

## Debug Mode

Click "Show debug info" at the bottom of the page to see detailed logs of speech recognition events. Useful for troubleshooting.

## Files

```
FrenchWhisper/
├── README.md         # This file
└── web/
    ├── index.html    # Complete app (HTML + CSS + JS)
    └── .gitignore
```

## License

MIT
