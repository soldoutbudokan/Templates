import Foundation
import Translation

@MainActor
final class TranslationService: ObservableObject {
    @Published var isReady = false
    @Published var isDownloading = false
    @Published var error: String?

    private var session: TranslationSession?
    private let sourceLanguage = Locale.Language(identifier: "fr")
    private let targetLanguage = Locale.Language(identifier: "en")

    var configuration: TranslationSession.Configuration {
        .init(source: sourceLanguage, target: targetLanguage)
    }

    func prepare(using session: TranslationSession) async {
        self.session = session
        isDownloading = true
        error = nil

        do {
            try await session.prepareTranslation()
            isDownloading = false
            isReady = true
        } catch {
            isDownloading = false
            self.error = "Failed to prepare translation: \(error.localizedDescription)"
        }
    }

    func translate(_ text: String) async -> String? {
        guard let session, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }

        do {
            let response = try await session.translate(text)
            return response.targetText
        } catch {
            self.error = "Translation error: \(error.localizedDescription)"
            return nil
        }
    }
}
