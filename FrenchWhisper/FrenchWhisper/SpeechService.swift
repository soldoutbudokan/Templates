import AVFoundation
import Speech

@MainActor
final class SpeechService: ObservableObject {
    @Published var isListening = false
    @Published var transcribedText = ""
    @Published var error: String?

    private var audioEngine: AVAudioEngine?
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private let speechRecognizer = SFSpeechRecognizer(locale: Locale(identifier: "fr-FR"))

    var onPartialResult: ((String) -> Void)?
    var onFinalResult: ((String) -> Void)?

    func requestPermissions() async -> Bool {
        let speechStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }

        guard speechStatus == .authorized else {
            error = "Speech recognition not authorized"
            return false
        }

        let audioStatus = await AVAudioApplication.requestRecordPermission()
        guard audioStatus else {
            error = "Microphone access not authorized"
            return false
        }

        return true
    }

    func startListening() async {
        guard let speechRecognizer, speechRecognizer.isAvailable else {
            error = "French speech recognition unavailable"
            return
        }

        guard speechRecognizer.supportsOnDeviceRecognition else {
            error = "On-device French recognition not supported"
            return
        }

        do {
            try await startAudioSession()
        } catch {
            self.error = "Failed to start audio: \(error.localizedDescription)"
            return
        }

        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest else {
            error = "Failed to create recognition request"
            return
        }

        recognitionRequest.shouldReportPartialResults = true
        recognitionRequest.requiresOnDeviceRecognition = true

        audioEngine = AVAudioEngine()
        guard let audioEngine else { return }

        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] buffer, _ in
            self?.recognitionRequest?.append(buffer)
        }

        recognitionTask = speechRecognizer.recognitionTask(with: recognitionRequest) { [weak self] result, error in
            Task { @MainActor in
                guard let self else { return }

                if let result {
                    let text = result.bestTranscription.formattedString
                    self.transcribedText = text

                    if result.isFinal {
                        self.onFinalResult?(text)
                    } else {
                        self.onPartialResult?(text)
                    }
                }

                if let error {
                    self.error = error.localizedDescription
                    self.stopListening()
                }
            }
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
            isListening = true
            error = nil
        } catch {
            self.error = "Audio engine failed: \(error.localizedDescription)"
            stopListening()
        }
    }

    func stopListening() {
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil

        recognitionRequest?.endAudio()
        recognitionRequest = nil

        recognitionTask?.cancel()
        recognitionTask = nil

        isListening = false
        transcribedText = ""
    }

    private func startAudioSession() async throws {
        #if os(macOS)
        // macOS doesn't require AVAudioSession configuration
        #else
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        #endif
    }
}
