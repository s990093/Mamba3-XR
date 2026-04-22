//
//  EyeView.swift
//  Mamba EYES
//
//  Cobra eye using a symmetric almond outline. Every concentric layer
//  (eyelid → socket → iris mask) uses the SAME shape at scaled sizes,
//  so geometry stays aligned and clipping never breaks.
//

import SwiftUI

struct EyeView: View {

    // Inputs
    let state: AppState
    let pitch: Double
    let roll: Double
    let thinkingOffset: CGFloat
    var mirrored: Bool = false

    /// How much to drop the INNER corner (toward the nose). 0 = symmetric.
    /// Try 0.12…0.20 for the reference "focused cobra" look.
    var innerDrop: CGFloat = 0.2
    /// How much to lift the OUTER corner. 0 = flat top.
    var outerLift: CGFloat = 0.02

    // Geometry (smaller than before)
    private let lidW: CGFloat = 240
    private let lidH: CGFloat = 190
    private let socketScale: CGFloat = 0.82
    private let irisScale: CGFloat = 0.72
    private let maxPupilOffset: CGFloat = 16

    @State private var speakingPulse = false
    @State private var idleBlink = false

    var body: some View {
        ZStack {
            // MARK: 1. Outer puffy eyelid
            AlmondShape(innerDrop: innerDrop, outerLift: outerLift)
                .fill(
                    LinearGradient(
                        colors: [Color(red: 0.22, green: 0.27, blue: 0.36),
                                 Color(red: 0.11, green: 0.14, blue: 0.22),
                                 Color(red: 0.06, green: 0.08, blue: 0.14)],
                        startPoint: .top, endPoint: .bottom
                    )
                )
                .frame(width: lidW, height: lidH)
                .overlay(
                    // Top rim highlight — fake 3D lip.
                    AlmondShape(innerDrop: innerDrop, outerLift: outerLift)
                        .trim(from: 0.05, to: 0.45)
                        .stroke(Color.white.opacity(0.2), lineWidth: 1.2)
                        .blur(radius: 0.5)
                        .frame(width: lidW, height: lidH)
                )
                .overlay(
                    // Sparse scale texture, masked to the eyelid.
                    ScaleTexture()
                        .stroke(Color.white.opacity(0.06), lineWidth: 0.6)
                        .frame(width: lidW, height: lidH)
                        .mask(AlmondShape(innerDrop: innerDrop, outerLift: outerLift).frame(width: lidW, height: lidH))
                )
                .shadow(color: .black.opacity(0.55), radius: 14, x: 0, y: 10)

            // MARK: 2. Inner socket (dark bezel)
            AlmondShape(innerDrop: innerDrop, outerLift: outerLift)
                .fill(
                    RadialGradient(
                        colors: [Color(red: 0.03, green: 0.04, blue: 0.07),
                                 Color(red: 0.07, green: 0.09, blue: 0.13)],
                        center: .center, startRadius: 10, endRadius: 120
                    )
                )
                .frame(width: lidW * socketScale, height: lidH * socketScale)
                .overlay(
                    AlmondShape(innerDrop: innerDrop, outerLift: outerLift)
                        .stroke(Color.black.opacity(0.9), lineWidth: 1.3)
                        .frame(width: lidW * socketScale, height: lidH * socketScale)
                )

            // MARK: 3. Iris — clipped to a smaller almond
            irisLayer
                .frame(width: lidW * irisScale, height: lidH * irisScale)
                .clipShape(AlmondShape(innerDrop: innerDrop, outerLift: outerLift))
                .scaleEffect(y: blinkScaleY, anchor: .center)
                .animation(.spring(response: 0.45, dampingFraction: 0.6), value: state)

            // MARK: 4. Pupil
            SlitPupilShape()
                .fill(
                    LinearGradient(
                        colors: [Color.black, Color(red: 0.04, green: 0.06, blue: 0.05)],
                        startPoint: .top, endPoint: .bottom
                    )
                )
                .frame(width: pupilWidth, height: pupilHeight)
                .offset(x: pupilOffsetX, y: pupilOffsetY)
                .scaleEffect(y: blinkScaleY, anchor: .center)
                .animation(.spring(response: 0.45, dampingFraction: 0.55), value: pupilOffsetX)
                .animation(.spring(response: 0.45, dampingFraction: 0.55), value: pupilOffsetY)
                .animation(.spring(response: 0.35, dampingFraction: 0.6), value: pupilWidth)
                .animation(.spring(response: 0.35, dampingFraction: 0.6), value: pupilHeight)

            // MARK: 5. Highlights
            Group {
                Circle()
                    .fill(Color.white)
                    .frame(width: 28, height: 28)
                    .shadow(color: .white.opacity(0.45), radius: 5)
                    .offset(x: pupilOffsetX - 34, y: pupilOffsetY - 26)

                Circle()
                    .fill(Color.white.opacity(0.85))
                    .frame(width: 10, height: 10)
                    .offset(x: pupilOffsetX - 52, y: pupilOffsetY + 4)
            }
            .scaleEffect(y: blinkScaleY, anchor: .center)

            // MARK: 6. Upper shadow — depth from upper eyelid
            AlmondShape(innerDrop: innerDrop, outerLift: outerLift)
                .fill(
                    LinearGradient(
                        colors: [Color.black.opacity(0.45),
                                 Color.black.opacity(0.15),
                                 Color.clear],
                        startPoint: .top, endPoint: .center
                    )
                )
                .frame(width: lidW * irisScale, height: lidH * irisScale)
                .allowsHitTesting(false)
                .scaleEffect(y: blinkScaleY, anchor: .center)
                .blendMode(.multiply)
        }
        .frame(width: lidW, height: lidH)
        .scaleEffect(x: mirrored ? -1 : 1, y: 1)
        .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true),
                   value: speakingPulse)
        .animation(.easeInOut(duration: 0.12), value: idleBlink)
        .onChange(of: state) { _, newValue in
            speakingPulse = (newValue == .speaking)
        }
        .onAppear { startIdleBlinkLoop() }
    }

    // MARK: - Iris (rendered as an oversized circle; almond clip above trims it)
    private var irisLayer: some View {
        ZStack {
            // Glow orb — warm core → lime → green → deep.
            Circle()
                .fill(
                    RadialGradient(
                        colors: [
                            Color(red: 1.00, green: 0.98, blue: 0.72),
                            Color(red: 0.78, green: 0.96, blue: 0.55),
                            Color(red: 0.40, green: 0.72, blue: 0.42),
                            Color(red: 0.12, green: 0.32, blue: 0.25),
                        ],
                        center: .init(x: 0.45, y: 0.55),
                        startRadius: 2, endRadius: 130
                    )
                )
                .scaleEffect(1.15) // oversize so clipping always covers the almond

            // Warm yellow pool under pupil.
            Ellipse()
                .fill(
                    RadialGradient(
                        colors: [Color(red: 1.0, green: 0.92, blue: 0.55).opacity(0.5),
                                 Color.clear],
                        center: .center, startRadius: 2, endRadius: 80
                    )
                )
                .frame(width: 160, height: 100)
                .offset(y: 28)
                .blendMode(.screen)

            // Iris fibre hatch.
            IrisHatchOverlay()
                .stroke(Color.white.opacity(0.06), lineWidth: 0.6)

            // Edge vignette.
            AlmondShape(innerDrop: innerDrop, outerLift: outerLift)
                .stroke(Color.black.opacity(0.55), lineWidth: 10)
                .blur(radius: 5)
        }
        .shadow(color: Color(red: 0.55, green: 1.0, blue: 0.6).opacity(
                    state == .speaking ? 0.5 : 0.28),
                radius: state == .speaking && speakingPulse ? 18 : 10)
    }

    // MARK: - Dynamic geometry
    private var pupilOffsetX: CGFloat {
        CGFloat(roll) * maxPupilOffset + thinkingOffset * maxPupilOffset * 0.9
    }
    private var pupilOffsetY: CGFloat { CGFloat(pitch) * maxPupilOffset }

    private var pupilWidth: CGFloat {
        switch state {
        case .listening: return 28
        case .thinking:  return 16
        default:         return 18
        }
    }
    private var pupilHeight: CGFloat {
        switch state {
        case .listening: return 130
        case .thinking:  return 115
        default:         return 122
        }
    }
    private var blinkScaleY: CGFloat { idleBlink ? 0.06 : 1.0 }

    private func startIdleBlinkLoop() {
        Task { @MainActor in
            while true {
                let wait = UInt64.random(in: 3_000_000_000...6_000_000_000)
                try? await Task.sleep(nanoseconds: wait)
                idleBlink = true
                try? await Task.sleep(nanoseconds: 120_000_000)
                idleBlink = false
            }
        }
    }
}

// MARK: - Shapes

/// Symmetric almond / lemon shape with optional asymmetric tilt.
///
/// Because the right eye is rendered with `.scaleEffect(x: -1)`, the RIGHT
/// side of this shape's local coordinates is always the "inner" corner
/// (closer to the nose). Tweak `innerDrop` / `outerLift` to make the eye
/// look angry, sad, sleepy, etc.
///
/// - `innerDrop`: 0…1, how far DOWN to push the inner (right) corner.
///                Positive values = inner corner lower than outer → more
///                "fierce / focused" look, like the reference image.
/// - `outerLift`: 0…1, how far UP to pull the outer (left) corner.
private struct AlmondShape: Shape {
    var innerDrop: CGFloat = 0
    var outerLift: CGFloat = 0

    func path(in rect: CGRect) -> Path {
        let w = rect.width, h = rect.height
        var p = Path()

        // Left = outer corner, Right = inner corner (see doc above).
        let outer = CGPoint(x: rect.minX + w * 0.02,
                            y: rect.midY - h * 0.5 * outerLift)
        let inner = CGPoint(x: rect.maxX - w * 0.02,
                            y: rect.midY + h * 0.5 * innerDrop)

        // Top/bottom control points pulled outward for rounded dome;
        // their Y shifts in sync with the corner tilt so curves stay smooth.
        let topCtrlL    = CGPoint(x: rect.minX + w * 0.18,
                                  y: rect.minY - h * 0.10 - h * 0.5 * outerLift)
        let topCtrlR    = CGPoint(x: rect.maxX - w * 0.18,
                                  y: rect.minY - h * 0.10 + h * 0.5 * innerDrop)
        let bottomCtrlR = CGPoint(x: rect.maxX - w * 0.18,
                                  y: rect.maxY + h * 0.10 + h * 0.5 * innerDrop)
        let bottomCtrlL = CGPoint(x: rect.minX + w * 0.18,
                                  y: rect.maxY + h * 0.10 - h * 0.5 * outerLift)

        p.move(to: outer)
        p.addCurve(to: inner, control1: topCtrlL, control2: topCtrlR)
        p.addCurve(to: outer, control1: bottomCtrlR, control2: bottomCtrlL)
        p.closeSubpath()
        return p
    }
}

/// Narrow pointed-top-and-bottom vertical slit pupil.
private struct SlitPupilShape: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let top    = CGPoint(x: rect.midX, y: rect.minY)
        let bottom = CGPoint(x: rect.midX, y: rect.maxY)
        let rightCtrl = CGPoint(x: rect.maxX + rect.width * 0.2, y: rect.midY)
        let leftCtrl  = CGPoint(x: rect.minX - rect.width * 0.2, y: rect.midY)
        p.move(to: top)
        p.addQuadCurve(to: bottom, control: rightCtrl)
        p.addQuadCurve(to: top,    control: leftCtrl)
        p.closeSubpath()
        return p
    }
}

/// Diagonal crosshatch — iris fibre lines.
private struct IrisHatchOverlay: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let step: CGFloat = 14
        var x: CGFloat = -rect.height
        while x < rect.width + rect.height {
            p.move(to: CGPoint(x: x, y: 0))
            p.addLine(to: CGPoint(x: x + rect.height, y: rect.height))
            p.move(to: CGPoint(x: x + rect.height, y: 0))
            p.addLine(to: CGPoint(x: x, y: rect.height))
            x += step
        }
        return p
    }
}

/// Hand-drawn scale strokes across the eyelid.
private struct ScaleTexture: Shape {
    func path(in rect: CGRect) -> Path {
        var p = Path()
        let cols = 14, rows = 8
        for r in 0..<rows {
            for c in 0..<cols {
                let x = rect.minX + CGFloat(c) * rect.width  / CGFloat(cols)
                let y = rect.minY + CGFloat(r) * rect.height / CGFloat(rows)
                let len: CGFloat = 6
                if (r + c) % 2 == 0 {
                    p.move(to: CGPoint(x: x, y: y))
                    p.addLine(to: CGPoint(x: x + len, y: y + len * 0.5))
                } else {
                    p.move(to: CGPoint(x: x, y: y))
                    p.addLine(to: CGPoint(x: x + len, y: y - len * 0.5))
                }
            }
        }
        return p
    }
}

#Preview {
    ZStack {
        LinearGradient(colors: [Color(red: 0.06, green: 0.09, blue: 0.16),
                                Color(red: 0.02, green: 0.04, blue: 0.09)],
                       startPoint: .top, endPoint: .bottom).ignoresSafeArea()
        HStack(spacing: 24) {
            EyeView(state: .idle, pitch: 0, roll: 0, thinkingOffset: 0, mirrored: false)
            EyeView(state: .idle, pitch: 0, roll: 0, thinkingOffset: 0, mirrored: true)
        }
    }
}
