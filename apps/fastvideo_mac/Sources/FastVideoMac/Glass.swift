import SwiftUI

// MARK: - Liquid Glass primitives
//
// Shared glass building blocks. On macOS 26 these render as native Liquid
// Glass; older systems receive a quiet material fallback so the app keeps its
// shape everywhere.

extension View {
    @ViewBuilder
    func fvLiquidGlass(
        cornerRadius: CGFloat = 16,
        tint: Color? = nil,
        interactive: Bool = false
    ) -> some View {
        if #available(macOS 26.0, *) {
            glassEffect(
                .regular.tint(tint).interactive(interactive),
                in: .rect(cornerRadius: cornerRadius)
            )
        } else {
            background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                        .stroke(Color.white.opacity(0.12), lineWidth: 1)
                )
        }
    }

    /// Content surfaces (settings cards, detail panels). Glass on macOS 26,
    /// a soft material card with a hairline stroke elsewhere.
    @ViewBuilder
    func fvGlassSurface(cornerRadius: CGFloat = 16, tint: Color? = nil) -> some View {
        if #available(macOS 26.0, *) {
            glassEffect(.regular.tint(tint), in: .rect(cornerRadius: cornerRadius))
        } else {
            background(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .fill(FVTheme.surface)
            )
            .overlay(
                RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
                    .stroke(FVTheme.line, lineWidth: 1)
            )
        }
    }

    /// Small pills and badges. Tinted glass capsule on macOS 26, flat capsule
    /// in `fallback` otherwise.
    @ViewBuilder
    func fvGlassCapsule(tint: Color? = nil, fallback: Color) -> some View {
        if #available(macOS 26.0, *) {
            glassEffect(.regular.tint(tint), in: .capsule)
        } else {
            background(Capsule().fill(fallback))
        }
    }
}

struct PlatformSidebarMaterial: View {
    var body: some View {
        if #available(macOS 26.0, *) {
            Rectangle()
                .fill(Color.black.opacity(0.12))
                .glassEffect(.regular.tint(FVTheme.sidebar.opacity(0.5)), in: .rect)
        } else {
            Rectangle().fill(FVTheme.sidebar)
        }
    }
}

// MARK: - Ambient aurora background
//
// A restrained, non-purple aurora: cool teal high on the trailing edge, warm
// sand low on the leading edge, and a breath of lime. It sits behind the ASCII
// field so hero surfaces feel alive while the video stays the focus.

struct FVAmbientBackground: View {
    var intensity: Double = 1
    var tint: Color = FVTheme.glowTeal
    var drift: Bool = false

    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var driftPhase = false

    var body: some View {
        ZStack {
            FVTheme.background
            RadialGradient(
                colors: [
                    tint.opacity(0.15 * intensity),
                    tint.opacity(0.045 * intensity),
                    .clear,
                ],
                center: .topTrailing,
                startRadius: 24,
                endRadius: 660
            )
            .scaleEffect(driftPhase ? 1.08 : 1)
            .opacity(driftPhase ? 0.82 : 1)
            RadialGradient(
                colors: [
                    FVTheme.glowWarm.opacity(0.10 * intensity),
                    .clear,
                ],
                center: UnitPoint(x: 0.06, y: 0.94),
                startRadius: 18,
                endRadius: 560
            )
            .scaleEffect(driftPhase ? 1.05 : 0.97)
            RadialGradient(
                colors: [
                    FVTheme.lime.opacity(0.05 * intensity),
                    .clear,
                ],
                center: UnitPoint(x: 0.24, y: 0.06),
                startRadius: 60,
                endRadius: 720
            )
        }
        .animation(.easeInOut(duration: 0.9), value: tint)
        .onAppear {
            guard drift, !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 6.5).repeatForever(autoreverses: true)) {
                driftPhase = true
            }
        }
    }
}
