//
//  AIServiceProtocol.swift
//  Mamba EYES
//
//  Abstraction over the AI backend. The ViewModel depends on this
//  protocol so we can freely swap `MockAIService` for a real HTTP /
//  on-device Mamba implementation later without touching the UI layer.
//

import Foundation

protocol AIServiceProtocol {
    /// Takes whatever audio/text input we have and returns the AI's reply.
    /// In the mock we ignore input and return a canned string after a delay.
    /// Later this signature can be extended (e.g. pass audio buffer, stream tokens).
    func processAudioInput() async throws -> String
}
