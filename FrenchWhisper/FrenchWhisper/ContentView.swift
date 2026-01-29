import SwiftUI
import Translation

struct ContentView: View {
    @StateObject private var speechService = SpeechService()
    @StateObject private var translationService = TranslationService()

    @State private var translatedText = ""
    @State private var statusMessage = "Ready"
    @State private var translationTask: Task<Void, Never>?

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            subtitleView

            Spacer()

            VStack(spacing: 16) {
                statusLabel
                controlButton
            }
            .padding(.bottom, 40)
        }
        .frame(minWidth: 600, minHeight: 400)
        .background(Color.black.opacity(0.85))
        .translationTask(translationService.configuration) { session in
            await translationService.prepare(using: session)
        }
        .task {
            await setupServices()
        }
        .onChange(of: translationService.isDownloading) { _, isDownloading in
            if isDownloading {
                statusMessage = "Downloading language pack..."
            } else if translationService.isReady && !speechService.isListening {
                statusMessage = "Ready"
            }
        }
        .onChange(of: speechService.isListening) { _, isListening in
            statusMessage = isListening ? "Listening..." : "Ready"
        }
    }

    private var subtitleView: some View {
        VStack(spacing: 12) {
            Text(translatedText.isEmpty ? "English subtitles will appear here" : translatedText)
                .font(.system(size: 32, weight: .medium))
                .foregroundColor(translatedText.isEmpty ? .gray : .white)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
                .padding(.vertical, 20)
                .background(
                    RoundedRectangle(cornerRadius: 12)
                        .fill(Color.black.opacity(0.6))
                )
                .animation(.easeInOut(duration: 0.15), value: translatedText)

            if !speechService.transcribedText.isEmpty {
                Text(speechService.transcribedText)
                    .font(.system(size: 18))
                    .foregroundColor(.gray)
                    .italic()
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }
        }
        .padding(.horizontal)
    }

    private var statusLabel: some View {
        HStack(spacing: 8) {
            if translationService.isDownloading || speechService.isListening {
                ProgressView()
                    .scaleEffect(0.7)
                    .progressViewStyle(.circular)
            }

            Text(statusMessage)
                .font(.system(size: 14))
                .foregroundColor(.secondary)
        }
    }

    private var controlButton: some View {
        Button(action: toggleListening) {
            HStack(spacing: 8) {
                Image(systemName: speechService.isListening ? "stop.fill" : "mic.fill")
                Text(speechService.isListening ? "Stop" : "Start")
            }
            .font(.system(size: 16, weight: .semibold))
            .foregroundColor(.white)
            .padding(.horizontal, 32)
            .padding(.vertical, 12)
            .background(
                RoundedRectangle(cornerRadius: 10)
                    .fill(speechService.isListening ? Color.red : Color.blue)
            )
        }
        .buttonStyle(.plain)
        .disabled(!translationService.isReady)
        .opacity(translationService.isReady ? 1.0 : 0.5)
    }

    private func setupServices() async {
        speechService.onPartialResult = { frenchText in
            scheduleTranslation(for: frenchText)
        }

        speechService.onFinalResult = { frenchText in
            scheduleTranslation(for: frenchText, immediate: true)
        }

        let granted = await speechService.requestPermissions()
        if !granted {
            statusMessage = speechService.error ?? "Permissions denied"
        }
    }

    private func scheduleTranslation(for text: String, immediate: Bool = false) {
        translationTask?.cancel()

        let delay: UInt64 = immediate ? 0 : 200_000_000

        translationTask = Task {
            if delay > 0 {
                try? await Task.sleep(nanoseconds: delay)
            }

            guard !Task.isCancelled else { return }

            if let translated = await translationService.translate(text) {
                await MainActor.run {
                    translatedText = translated
                }
            }
        }
    }

    private func toggleListening() {
        if speechService.isListening {
            speechService.stopListening()
        } else {
            translatedText = ""
            Task {
                await speechService.startListening()
            }
        }
    }
}

#Preview {
    ContentView()
}
