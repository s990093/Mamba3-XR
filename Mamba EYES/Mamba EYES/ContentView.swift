//
//  ContentView.swift
//  Mamba EYES
//
//  Full-screen, LANDSCAPE Cobra Eye companion.
//  Interaction: tap anywhere on the screen to toggle listening.
//  (Will be replaced by wake-word / VAD-driven STT later.)
//

import SwiftUI

struct ContentView: View {

    @StateObject private var viewModel = EyesViewModel()
    @StateObject private var motion = MotionService()

    var body: some View {
        ZStack {
            // Dark gradient backdrop (deep navy, matches reference art).
            LinearGradient(
                colors: [Color(red: 0.06, green: 0.09, blue: 0.16),
                         Color(red: 0.02, green: 0.04, blue: 0.09)],
                startPoint: .top,
                endPoint: .bottom
            )
            .ignoresSafeArea()

            // Two cobra eyes, centered.
            HStack(spacing: 24) {
                EyeView(state: viewModel.state,
                        pitch: motion.normalizedPitch,
                        roll: motion.normalizedRoll,
                        thinkingOffset: viewModel.thinkingOffset,
                        mirrored: false)

                EyeView(state: viewModel.state,
                        pitch: motion.normalizedPitch,
                        roll: motion.normalizedRoll,
                        thinkingOffset: viewModel.thinkingOffset,
                        mirrored: true)
            }

            // HUD overlay (status + last response), non-intrusive.
            VStack {
                HStack {
                    Text(statusText)
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.cyan.opacity(0.7))
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Capsule().fill(.black.opacity(0.4)))
                    Spacer()
                    Text("tap to talk")
                        .font(.system(.caption2, design: .monospaced))
                        .foregroundStyle(.white.opacity(0.35))
                }
                .padding(.horizontal, 24)
                .padding(.top, 16)

                Spacer()

                if !viewModel.lastResponse.isEmpty {
                    Text(viewModel.lastResponse)
                        .font(.callout)
                        .foregroundStyle(.white.opacity(0.75))
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 48)
                        .padding(.bottom, 24)
                        .transition(.opacity)
                }
            }
        }
        // Whole-screen tap → toggle listening.
        .contentShape(Rectangle())
        .onTapGesture { handleTap() }
        .onAppear  { motion.start() }
        .onDisappear { motion.stop() }
        .statusBarHidden(true)
        .persistentSystemOverlays(.hidden)
    }

    // MARK: - Actions

    private func handleTap() {
        switch viewModel.state {
        case .idle:
            viewModel.startListening()
        case .listening:
            viewModel.stopListeningAndProcess()
        case .thinking, .speaking:
            break // ignore taps mid-pipeline
        }
    }

    // MARK: - HUD helpers

    private var statusText: String {
        switch viewModel.state {
        case .idle:      return "● IDLE"
        case .listening: return "● LISTENING…"
        case .thinking:  return "● THINKING…"
        case .speaking:  return "● SPEAKING…"
        }
    }
}

#Preview {
    ContentView()
}
