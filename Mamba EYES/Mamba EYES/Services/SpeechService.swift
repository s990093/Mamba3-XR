//
//  SpeechService.swift
//  Mamba EYES
//
//  Wraps AVSpeechSynthesizer so the ViewModel can `await` a spoken
//  utterance and receive a callback when speech finishes.
//

import Foundation
import AVFoundation

@MainActor
final class SpeechService: NSObject, ObservableObject {

    private let synthesizer = AVSpeechSynthesizer()
    private var finishContinuation: CheckedContinuation<Void, Never>?

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Speak the given text and suspend until playback is finished.
    func speak(_ text: String, language: String = "en-US") async {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: language)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            self.finishContinuation = cont
            synthesizer.speak(utterance)
        }
    }

    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
    }
}

extension SpeechService: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.finishContinuation?.resume()
            self.finishContinuation = nil
        }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer,
                                       didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in
            self.finishContinuation?.resume()
            self.finishContinuation = nil
        }
    }
}
