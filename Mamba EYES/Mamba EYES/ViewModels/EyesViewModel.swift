//
//  EyesViewModel.swift
//  Mamba EYES
//
//  Drives the Cobra's state machine and exposes UI-bindable properties.
//

import Foundation
import SwiftUI

@MainActor
final class EyesViewModel: ObservableObject {

    // MARK: - Published UI state

    @Published private(set) var state: AppState = .idle
    @Published private(set) var lastResponse: String = ""

    /// Horizontal pupil offset (-1…1) used by the `thinking` dart animation.
    @Published var thinkingOffset: CGFloat = 0

    // MARK: - Dependencies (injected → swap mock for real later)

    private let aiService: AIServiceProtocol
    private let speechService: SpeechService

    private var thinkingTask: Task<Void, Never>?

    init(aiService: AIServiceProtocol? = nil,
         speechService: SpeechService? = nil) {
        // Build defaults inside the MainActor-isolated init body rather than
        // in default argument expressions (which are evaluated in a
        // nonisolated context and would fail to construct MainActor types).
        self.aiService = aiService ?? MockAIService()
        self.speechService = speechService ?? SpeechService()
    }

    // MARK: - Public intents called by the View

    /// Called when the user starts holding / taps the talk button.
    func startListening() {
        guard state == .idle else { return }
        state = .listening
    }

    /// Called when the user releases / taps again to stop talking.
    func stopListeningAndProcess() {
        guard state == .listening else { return }
        Task { await runPipeline() }
    }

    // MARK: - Pipeline

    private func runPipeline() async {
        // 1) Thinking phase — kick off AI call + darting animation.
        state = .thinking
        startDartingAnimation()

        let reply: String
        do {
            reply = try await aiService.processAudioInput()
        } catch {
            reply = "Something went wrong. Try again."
        }

        stopDartingAnimation()
        lastResponse = reply

        // 2) Speaking phase.
        state = .speaking
        await speechService.speak(reply)

        // 3) Back to idle.
        state = .idle
    }

    // MARK: - Thinking animation helper

    private func startDartingAnimation() {
        thinkingTask?.cancel()
        thinkingTask = Task { [weak self] in
            guard let self else { return }
            var direction: CGFloat = 1
            while !Task.isCancelled {
                await MainActor.run {
                    withAnimation(.easeInOut(duration: 0.35)) {
                        self.thinkingOffset = direction
                    }
                }
                direction *= -1
                try? await Task.sleep(nanoseconds: 400_000_000)
            }
        }
    }

    private func stopDartingAnimation() {
        thinkingTask?.cancel()
        thinkingTask = nil
        withAnimation(.spring()) {
            thinkingOffset = 0
        }
    }
}
