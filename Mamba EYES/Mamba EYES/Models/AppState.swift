//
//  AppState.swift
//  Mamba EYES
//
//  Finite state machine describing what the Cobra is currently doing.
//

import Foundation

/// The four lifecycle states the Cobra goes through during a single
/// user interaction loop: idle → listening → thinking → speaking → idle.
enum AppState: Equatable {
    case idle
    case listening
    case thinking
    case speaking
}
