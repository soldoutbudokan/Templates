# FrenchWhisper

A macOS app that provides live French-to-English audio translation, running entirely on-device.

## Requirements

- **macOS 15.0 (Sequoia)** or later
- **Xcode 15** or later
- Apple Silicon or Intel Mac with on-device speech recognition support

## Features

- Real-time French speech recognition using Apple's Speech framework
- On-device translation using Apple's Translation framework
- Live streaming subtitles that update as you speak
- 100% on-device processing - no network calls required
- Dark semi-transparent UI optimized for subtitle display

## Building

1. Open `FrenchWhisper.xcodeproj` in Xcode
2. Select your development team in Signing & Capabilities
3. Build and run (Cmd+R)

## First Run

On first launch, the app will:

1. Request microphone permission
2. Request speech recognition permission
3. Download the French language pack for translation (if not already installed)

The status label will show "Downloading language pack..." while the Translation framework downloads the required on-device models.

## Usage

1. Click **Start** to begin listening
2. Speak French into your microphone
3. Watch English subtitles appear in real-time
4. Click **Stop** when done

The French transcription appears in smaller gray text below the English translation.

## Architecture

```
FrenchWhisper/
├── FrenchWhisperApp.swift      # App entry point
├── ContentView.swift           # Main UI
├── SpeechService.swift         # Mic capture + French transcription
└── TranslationService.swift    # French→English translation
```

## Technical Notes

- Uses `SFSpeechRecognizer` with `fr-FR` locale and `requiresOnDeviceRecognition = true`
- Uses SwiftUI's `translationTask` modifier to manage the Translation session
- Implements debouncing (200ms) for partial results to avoid overwhelming the translator
- Audio is captured via `AVAudioEngine` with a tap on the input node

## Permissions

The app requires these permissions (configured via Info.plist keys in build settings):

- `NSMicrophoneUsageDescription` - Microphone access for audio capture
- `NSSpeechRecognitionUsageDescription` - Speech recognition for French transcription

## License

MIT
