//
//  MotionService.swift
//  Mamba EYES
//
//  Thin wrapper around CoreMotion that publishes normalized pitch/roll
//  values in the range [-1, 1], ready to be multiplied by a pixel offset.
//

import Foundation
import CoreMotion
import Combine

@MainActor
final class MotionService: ObservableObject {

    /// Normalized roll  (left / right tilt), range roughly [-1, 1].
    @Published var normalizedRoll: Double = 0
    /// Normalized pitch (forward / backward tilt), range roughly [-1, 1].
    @Published var normalizedPitch: Double = 0

    private let manager = CMMotionManager()
    private let queue = OperationQueue()

    /// How far (in radians) we consider "fully tilted" before clamping.
    /// ~45° feels natural for a desk companion.
    private let maxTilt: Double = .pi / 4

    func start() {
        guard manager.isDeviceMotionAvailable else { return }
        manager.deviceMotionUpdateInterval = 1.0 / 60.0
        manager.startDeviceMotionUpdates(to: queue) { [weak self] motion, _ in
            guard let self, let motion else { return }
            let roll  = max(-1, min(1, motion.attitude.roll  / self.maxTilt))
            let pitch = max(-1, min(1, motion.attitude.pitch / self.maxTilt))
            Task { @MainActor in
                self.normalizedRoll = roll
                self.normalizedPitch = pitch
            }
        }
    }

    func stop() {
        manager.stopDeviceMotionUpdates()
    }

    deinit {
        manager.stopDeviceMotionUpdates()
    }
}
