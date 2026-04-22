//
//  MockAIService.swift
//  Mamba EYES
//
//  Fake AI backend used for the demo. Simulates network + model latency
//  with `Task.sleep` and returns a random canned response.
//

import Foundation

final class MockAIService: AIServiceProtocol {

    private let cannedReplies: [String] = [
        "I am the Cobra. I have analyzed your request, but honestly, I just want a donut.",
        "My neural weights say yes. My heart says… also yes, but with sprinkles.",
        "Processing complete. Recommendation: take a break, hydrate, and pet a cat.",
        "I could answer that in 400 billion parameters, or I could just say: probably.",
        "Affirmative, human. Your request has been filed under 'interesting but weird'."
    ]

    func processAudioInput() async throws -> String {
        // Simulate STT + model inference + TTS synthesis latency.
        try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds
        return cannedReplies.randomElement() ?? cannedReplies[0]
    }
}
