import SwiftUI

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

struct OnboardingView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var page = 0
    @Namespace private var glassNamespace

    let onFinish: (AppSection) -> Void

    private let pageCount = 4

    var body: some View {
        ZStack {
            FVTheme.background
                .ignoresSafeArea()
            ASCIIFieldView()
                .opacity(0.38)
                .mask(
                    RadialGradient(
                        colors: [.white, .white.opacity(0.55), .clear],
                        center: .topTrailing,
                        startRadius: 20,
                        endRadius: 720
                    )
                )
                .ignoresSafeArea()
            LinearGradient(
                colors: [FVTheme.background.opacity(0.08), FVTheme.background.opacity(0.72)],
                startPoint: .topTrailing,
                endPoint: .bottomLeading
            )
            .ignoresSafeArea()

            VStack(spacing: 0) {
                onboardingHeader
                ZStack {
                    pageContent
                        .id(page)
                        .transition(pageTransition)
                }
                .animation(reduceMotion ? nil : .spring(response: 0.62, dampingFraction: 0.88), value: page)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                onboardingControls
            }
            .padding(28)
        }
        .foregroundStyle(FVTheme.text)
    }

    private var onboardingHeader: some View {
        HStack(spacing: 12) {
            HStack(spacing: 9) {
                ZStack {
                    RoundedRectangle(cornerRadius: 6, style: .continuous)
                        .fill(FVTheme.lime)
                    Text("FQ")
                        .font(.system(size: 10, weight: .black, design: .monospaced))
                        .foregroundStyle(.black)
                }
                .frame(width: 30, height: 30)
                Text("FASTWAN QAD")
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .tracking(1.4)
            }
            Spacer()
            HStack(spacing: 7) {
                ForEach(0..<pageCount, id: \.self) { index in
                    Capsule()
                        .fill(index == page ? FVTheme.lime : Color.white.opacity(0.16))
                        .frame(width: index == page ? 28 : 8, height: 6)
                        .animation(reduceMotion ? nil : .spring(response: 0.4), value: page)
                }
            }
            .padding(.horizontal, 13)
            .frame(height: 32)
            .fvLiquidGlass(cornerRadius: 16)
            Spacer()
            Button("Skip") { onFinish(.create) }
                .buttonStyle(.plain)
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(FVTheme.muted)
                .padding(.horizontal, 12)
                .frame(height: 32)
                .fvLiquidGlass(cornerRadius: 16, interactive: true)
        }
    }

    @ViewBuilder
    private var pageContent: some View {
        switch page {
        case 0: welcomePage
        case 1: previewPage
        case 2: localPage
        default: readyPage
        }
    }

    private var welcomePage: some View {
        HStack(spacing: 58) {
            VStack(alignment: .leading, spacing: 0) {
                OnboardingEyebrow(text: "WELCOME TO FASTWAN QAD")
                Text("A video model that\nlives on your Mac.")
                    .font(.system(size: 55, weight: .medium, design: .rounded))
                    .tracking(-2.5)
                    .lineSpacing(-4)
                    .padding(.top, 17)
                Text("No queue. No API key. No prompt leaving the machine. FastWan QAD runs through MLX and Metal, then gives you a real MP4.")
                    .font(.system(size: 15))
                    .foregroundStyle(FVTheme.muted)
                    .frame(maxWidth: 530, alignment: .leading)
                    .padding(.top, 20)

                VStack(alignment: .leading, spacing: 9) {
                    Text("PICK A FIRST DIRECTION")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .tracking(1.2)
                        .foregroundStyle(FVTheme.faint)
                    HStack(spacing: 8) {
                        PromptSeedButton(label: "Misty forest", prompt: "A fox runs through a misty pine forest, leaves kicking up behind it.")
                        PromptSeedButton(label: "Neon rain", prompt: "A lone tram glides through neon rain at midnight, reflections stretching across wet asphalt.")
                        PromptSeedButton(label: "Paper world", prompt: "A paper boat sails through a miniature city made from folded maps, warm afternoon light.")
                    }
                }
                .padding(.top, 30)
            }
            Spacer(minLength: 20)
            OnboardingModelObject()
                .frame(width: 340, height: 340)
        }
        .padding(.horizontal, 48)
        .frame(maxWidth: 1180)
    }

    private var previewPage: some View {
        HStack(spacing: 54) {
            VStack(alignment: .leading, spacing: 0) {
                OnboardingEyebrow(text: "LIVE X0 PREVIEW")
                Text("The render starts\nbefore it ends.")
                    .font(.system(size: 52, weight: .medium, design: .rounded))
                    .tracking(-2.2)
                    .lineSpacing(-4)
                    .padding(.top, 17)
                Text("DMD predicts the whole video at every step. After step one, TAEHV turns that rough prediction into a playable clip while MLX keeps refining.")
                    .font(.system(size: 15))
                    .foregroundStyle(FVTheme.muted)
                    .frame(maxWidth: 510, alignment: .leading)
                    .padding(.top, 20)
                HStack(spacing: 26) {
                    OnboardingMetric(value: "~40s", label: "FIRST LIGHT")
                    OnboardingMetric(value: "1–2s", label: "PREVIEW DECODE")
                    OnboardingMetric(value: "3", label: "DMD STEPS")
                }
                .padding(.top, 29)
            }
            Spacer(minLength: 12)
            PreviewTimelineDemo()
                .frame(width: 460)
        }
        .padding(.horizontal, 50)
        .frame(maxWidth: 1180)
    }

    private var localPage: some View {
        VStack(spacing: 35) {
            VStack(spacing: 14) {
                OnboardingEyebrow(text: "YOURS MEANS YOURS")
                Text("Private by architecture.")
                    .font(.system(size: 49, weight: .medium, design: .rounded))
                    .tracking(-2)
                Text("Prompts and videos stay on this Mac. The model runs locally through MLX and Metal.")
                    .font(.system(size: 14.5))
                    .foregroundStyle(FVTheme.muted)
            }
            HStack(spacing: 12) {
                OnboardingFeature(
                    symbol: "lock.shield",
                    title: "Zero uploads",
                    detail: "Prompts, latents, previews, and final MP4s stay on this Mac."
                )
                OnboardingFeature(
                    symbol: "film.stack",
                    title: "A durable library",
                    detail: "Every shot remembers its prompt, seed, settings, timing, and output."
                )
                OnboardingFeature(
                    symbol: "square.and.arrow.up",
                    title: "Native sharing",
                    detail: "Export to Finder or hand the finished video to the macOS Share Sheet."
                )
            }
            .frame(maxWidth: 970)
        }
        .padding(.horizontal, 50)
    }

    private var readyPage: some View {
        HStack(spacing: 58) {
            VStack(alignment: .leading, spacing: 0) {
                OnboardingEyebrow(text: "ONE LAST THING")
                Text("Ready the machine.\nThen make the shot.")
                    .font(.system(size: 50, weight: .medium, design: .rounded))
                    .tracking(-2.1)
                    .lineSpacing(-3)
                    .padding(.top, 16)
                Text("Install the recommended EMA model with one click. FastWan QAD manages the model, runtime, and video exporter for you.")
                    .font(.system(size: 14.5))
                    .foregroundStyle(FVTheme.muted)
                    .frame(maxWidth: 520, alignment: .leading)
                    .padding(.top, 20)
                if model.runtimeHealth.emaAvailable {
                    Label("EMA model installed", systemImage: "checkmark.circle.fill")
                        .font(.system(size: 13, weight: .medium))
                        .foregroundStyle(FVTheme.lime)
                        .padding(.top, 26)
                } else if model.isInstallingRuntime {
                    VStack(alignment: .leading, spacing: 8) {
                        ProgressView().controlSize(.small)
                        Text("Preparing the local runtime…")
                            .font(.system(size: 11.5))
                            .foregroundStyle(FVTheme.muted)
                    }
                    .padding(.top, 26)
                } else if model.installingVariant == .ema {
                    VStack(alignment: .leading, spacing: 8) {
                        ProgressView(value: model.modelInstallProgress)
                            .tint(FVTheme.lime)
                            .frame(width: 300)
                        Text("Downloading the EMA model…")
                            .font(.system(size: 11.5))
                            .foregroundStyle(FVTheme.muted)
                    }
                    .padding(.top, 26)
                } else {
                    Button {
                        model.installModelWithRuntime(.ema)
                    } label: {
                        Label("Download EMA model", systemImage: "arrow.down.circle.fill")
                            .font(.system(size: 13, weight: .semibold))
                            .foregroundStyle(Color.black.opacity(0.84))
                            .padding(.horizontal, 18)
                            .frame(height: 42)
                            .background(Capsule().fill(FVTheme.lime))
                    }
                    .buttonStyle(.plain)
                    .padding(.top, 26)
                }
            }
            Spacer(minLength: 12)
            VStack(spacing: 9) {
                OnboardingCheckRow(
                    symbol: "apple.logo",
                    label: "Apple Silicon",
                    detail: model.runtimeHealth.platformSupported ? "Detected" : "Required",
                    ready: model.runtimeHealth.platformSupported
                )
                OnboardingCheckRow(
                    symbol: "cpu",
                    label: "MLX + Metal runtime",
                    detail: model.runtimeHealth.mlxAvailable && model.runtimeHealth.mpsAvailable ? "Ready" : "Install in Models & Runtime",
                    ready: model.runtimeHealth.mlxAvailable && model.runtimeHealth.mpsAvailable
                )
                OnboardingCheckRow(
                    symbol: "film",
                    label: "FastWan QAD 1.3B",
                    detail: model.runtimeHealth.emaAvailable ? "EMA installed · RAW optional" : "Download EMA to begin",
                    ready: model.runtimeHealth.rawAvailable || model.runtimeHealth.emaAvailable
                )
                OnboardingCheckRow(
                    symbol: "shippingbox",
                    label: "ffmpeg export",
                    detail: model.runtimeHealth.ffmpegAvailable ? "Ready" : "Install in Models & Runtime",
                    ready: model.runtimeHealth.ffmpegAvailable
                )
            }
            .padding(18)
            .frame(width: 420)
            .fvLiquidGlass(cornerRadius: 22, tint: FVTheme.lime.opacity(0.035))
        }
        .padding(.horizontal, 52)
        .frame(maxWidth: 1180)
    }

    @ViewBuilder
    private var onboardingControls: some View {
        if #available(macOS 26.0, *) {
            GlassEffectContainer(spacing: 12) {
                controlsContent(useNativeGlass: true)
            }
        } else {
            controlsContent(useNativeGlass: false)
        }
    }

    private func controlsContent(useNativeGlass: Bool) -> some View {
        HStack(spacing: 10) {
            if page > 0 {
                onboardingButton(title: "Back", symbol: "chevron.left", prominent: false, useNativeGlass: useNativeGlass) {
                    withAnimation { page -= 1 }
                }
            }
            Spacer()
            if page == pageCount - 1 {
                if model.runtimeHealth.emaAvailable {
                    onboardingButton(title: "Start creating", symbol: "arrow.right", prominent: true, useNativeGlass: useNativeGlass) {
                        onFinish(.create)
                    }
                } else {
                    onboardingButton(title: "Open Models & Runtime", symbol: "arrow.right", prominent: true, useNativeGlass: useNativeGlass) {
                        onFinish(.setup)
                    }
                }
            } else {
                onboardingButton(title: "Continue", symbol: "arrow.right", prominent: true, useNativeGlass: useNativeGlass) {
                    withAnimation { page += 1 }
                }
            }
        }
        .frame(maxWidth: 1080)
        .padding(.horizontal, 28)
        .padding(.bottom, 6)
    }

    @ViewBuilder
    private func onboardingButton(
        title: String,
        symbol: String?,
        prominent: Bool,
        useNativeGlass: Bool,
        action: @escaping () -> Void
    ) -> some View {
        let button = Button(action: action) {
            HStack(spacing: 8) {
                if symbol == "chevron.left" { Image(systemName: symbol!) }
                Text(title)
                if let symbol, symbol != "chevron.left" { Image(systemName: symbol) }
            }
            .font(.system(size: 12, weight: .semibold))
            .padding(.horizontal, 18)
            .frame(height: 40)
        }
        if #available(macOS 26.0, *), useNativeGlass {
            if prominent {
                button.buttonStyle(.glassProminent).tint(FVTheme.lime)
            } else {
                button.buttonStyle(.glass)
            }
        } else {
            button
                .buttonStyle(.plain)
                .foregroundStyle(prominent ? Color.black : FVTheme.text)
                .background(
                    Capsule().fill(prominent ? FVTheme.lime : Color.white.opacity(0.08))
                )
                .overlay(Capsule().stroke(Color.white.opacity(prominent ? 0 : 0.12)))
        }
    }

    private var pageTransition: AnyTransition {
        reduceMotion ? .opacity : .asymmetric(
            insertion: .move(edge: .trailing).combined(with: .opacity),
            removal: .move(edge: .leading).combined(with: .opacity)
        )
    }
}

private struct OnboardingEyebrow: View {
    let text: String
    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(FVTheme.lime).frame(width: 6, height: 6)
            Text(text)
                .font(.system(size: 9.5, weight: .bold, design: .monospaced))
                .tracking(1.25)
                .foregroundStyle(FVTheme.muted)
        }
    }
}

private struct PromptSeedButton: View {
    @EnvironmentObject private var model: AppModel
    let label: String
    let prompt: String

    var body: some View {
        Button(label) { model.prompt = prompt }
            .buttonStyle(.plain)
            .font(.system(size: 10.5, weight: .medium))
            .foregroundStyle(FVTheme.muted)
            .padding(.horizontal, 12)
            .frame(height: 32)
            .fvLiquidGlass(cornerRadius: 16, interactive: true)
    }
}

private struct OnboardingModelObject: View {
    @State private var animate = false
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        ZStack {
            ForEach(0..<5, id: \.self) { index in
                RoundedRectangle(cornerRadius: 28 + CGFloat(index) * 4, style: .continuous)
                    .stroke(
                        index == 0 ? FVTheme.lime.opacity(0.7) : Color.white.opacity(0.08 + Double(index) * 0.025),
                        lineWidth: index == 0 ? 1.5 : 1
                    )
                    .frame(width: 126 + CGFloat(index) * 38, height: 126 + CGFloat(index) * 38)
                    .rotationEffect(.degrees((animate ? 14 : -14) * Double(index.isMultiple(of: 2) ? 1 : -1)))
            }
            VStack(spacing: 8) {
                Text("1.3B")
                    .font(.system(size: 39, weight: .medium, design: .monospaced))
                Text("LOCAL PARAMETERS")
                    .font(.system(size: 8, weight: .bold, design: .monospaced))
                    .tracking(1.1)
                    .foregroundStyle(FVTheme.faint)
            }
        }
        .fvLiquidGlass(cornerRadius: 42, tint: FVTheme.lime.opacity(0.025))
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 4.5).repeatForever(autoreverses: true)) { animate = true }
        }
    }
}

private struct OnboardingMetric: View {
    let value: String
    let label: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value).font(.system(size: 20, weight: .medium, design: .monospaced))
            Text(label)
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .tracking(1)
                .foregroundStyle(FVTheme.faint)
        }
    }
}

private struct PreviewTimelineDemo: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                HStack(spacing: 6) {
                    Circle().fill(FVTheme.lime).frame(width: 6, height: 6)
                    Text("STREAMING LOCALLY")
                        .font(.system(size: 8.5, weight: .bold, design: .monospaced))
                        .tracking(1)
                }
                Spacer()
                Text("DMD / 03")
                    .font(.system(size: 8.5, design: .monospaced))
                    .foregroundStyle(FVTheme.faint)
            }
            ZStack {
                RoundedRectangle(cornerRadius: 13).fill(Color.black.opacity(0.36))
                Text("··::--==++**##\n·::--==++**##%\n::--==++**##%@\n:--==++**##%@%\n--==++**##%@%#")
                    .font(.system(size: 15, design: .monospaced))
                    .tracking(4)
                    .lineSpacing(5)
                    .foregroundStyle(FVTheme.lime.opacity(0.55))
                    .blur(radius: 0.2)
                VStack {
                    Spacer()
                    HStack {
                        Text("ROUGH X0 / STEP 1")
                            .font(.system(size: 8, weight: .bold, design: .monospaced))
                            .padding(.horizontal, 8)
                            .frame(height: 24)
                            .fvLiquidGlass(cornerRadius: 12)
                        Spacer()
                    }
                    .padding(11)
                }
            }
            .aspectRatio(832.0 / 480.0, contentMode: .fit)
            HStack(spacing: 6) {
                TimelineStep(label: "01", state: "PREVIEW", active: true)
                TimelineStep(label: "02", state: "REFINING", active: false)
                TimelineStep(label: "03", state: "FINAL", active: false)
            }
        }
        .padding(17)
        .fvLiquidGlass(cornerRadius: 24, tint: FVTheme.lime.opacity(0.035))
    }
}

private struct TimelineStep: View {
    let label: String
    let state: String
    let active: Bool
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(label)
                Spacer()
                Circle().fill(active ? FVTheme.lime : Color.white.opacity(0.15)).frame(width: 5, height: 5)
            }
            Text(state).foregroundStyle(active ? FVTheme.text : FVTheme.faint)
        }
        .font(.system(size: 8, weight: .bold, design: .monospaced))
        .padding(9)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.18)))
    }
}

private struct OnboardingFeature: View {
    let symbol: String
    let title: String
    let detail: String
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Image(systemName: symbol)
                .font(.system(size: 18, weight: .medium))
                .foregroundStyle(FVTheme.lime)
                .frame(width: 42, height: 42)
                .fvLiquidGlass(cornerRadius: 14, tint: FVTheme.lime.opacity(0.05))
            Text(title).font(.system(size: 15, weight: .semibold))
            Text(detail)
                .font(.system(size: 11.5))
                .foregroundStyle(FVTheme.muted)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(20)
        .frame(maxWidth: .infinity, minHeight: 190, alignment: .topLeading)
        .background(RoundedRectangle(cornerRadius: 18).fill(Color.white.opacity(0.04)))
        .overlay(RoundedRectangle(cornerRadius: 18).stroke(Color.white.opacity(0.09)))
    }
}

private struct OnboardingCheckRow: View {
    let symbol: String
    let label: String
    let detail: String
    let ready: Bool
    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: symbol)
                .font(.system(size: 13, weight: .medium))
                .frame(width: 28, height: 28)
                .background(Circle().fill(Color.white.opacity(0.07)))
            VStack(alignment: .leading, spacing: 2) {
                Text(label).font(.system(size: 11.5, weight: .semibold))
                Text(detail).font(.system(size: 9.5)).foregroundStyle(FVTheme.faint)
            }
            Spacer()
            Image(systemName: ready ? "checkmark.circle.fill" : "circle.dotted")
                .foregroundStyle(ready ? FVTheme.lime : FVTheme.amber)
        }
        .padding(.horizontal, 12)
        .frame(height: 54)
        .background(RoundedRectangle(cornerRadius: 11).fill(Color.black.opacity(0.14)))
    }
}
