import AVKit
import SwiftUI

enum FVTheme {
    static let background = Color(red: 0.027, green: 0.031, blue: 0.028)
    static let sidebar = Color(red: 0.045, green: 0.051, blue: 0.046)
    static let surface = Color.white.opacity(0.045)
    static let surfaceStrong = Color.white.opacity(0.075)
    static let line = Color.white.opacity(0.105)
    static let text = Color.white.opacity(0.94)
    static let muted = Color.white.opacity(0.52)
    static let faint = Color.white.opacity(0.28)
    static let lime = Color(red: 0.72, green: 1.0, blue: 0.42)
    static let amber = Color(red: 1.0, green: 0.73, blue: 0.34)
    static let red = Color(red: 1.0, green: 0.38, blue: 0.32)
}

struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("fastvideo.onboarding.completed") private var onboardingCompleted = false

    var body: some View {
        Group {
            if onboardingCompleted {
                HStack(spacing: 0) {
                    SidebarView()
                        .frame(width: 184)
                    Rectangle()
                        .fill(FVTheme.line)
                        .frame(width: 1)
                    Group {
                        switch model.section {
                        case .create: CreateView()
                        case .library: LibraryView()
                        case .setup: SetupView()
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                OnboardingView { destination in
                    model.section = destination
                    onboardingCompleted = true
                }
            }
        }
        .background(FVTheme.background)
        .preferredColorScheme(.dark)
        .alert(
            "FastVideo",
            isPresented: Binding(
                get: { model.alertMessage != nil },
                set: { if !$0 { model.alertMessage = nil } }
            )
        ) {
            Button("OK") { model.alertMessage = nil }
        } message: {
            Text(model.alertMessage ?? "")
        }
    }
}

private struct SidebarView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 10) {
                ZStack {
                    RoundedRectangle(cornerRadius: 5)
                        .fill(FVTheme.lime)
                    Text("FV")
                        .font(.system(size: 10, weight: .black, design: .monospaced))
                        .foregroundStyle(.black)
                }
                .frame(width: 28, height: 28)
                VStack(alignment: .leading, spacing: 1) {
                    Text("FASTVIDEO")
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                        .tracking(1.4)
                    Text("APPLE / MLX")
                        .font(.system(size: 8, weight: .medium, design: .monospaced))
                        .foregroundStyle(FVTheme.faint)
                        .tracking(1.1)
                }
            }
            .padding(.top, 38)
            .padding(.horizontal, 20)
            .padding(.bottom, 34)

            VStack(spacing: 6) {
                ForEach(AppSection.allCases) { section in
                    Button {
                        model.section = section
                    } label: {
                        HStack(spacing: 11) {
                            Image(systemName: section.symbol)
                                .font(.system(size: 13, weight: .medium))
                                .frame(width: 18)
                            Text(section.label)
                                .font(.system(size: 13, weight: .medium))
                            Spacer()
                            if section == .library, !model.records.isEmpty {
                                Text("\(model.records.count)")
                                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                                    .foregroundStyle(FVTheme.faint)
                            }
                        }
                        .foregroundStyle(model.section == section ? FVTheme.text : FVTheme.muted)
                        .padding(.horizontal, 12)
                        .frame(height: 38)
                        .background(
                            RoundedRectangle(cornerRadius: 7)
                                .fill(model.section == section ? Color.white.opacity(0.075) : .clear)
                        )
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 9)

            Spacer()

            VStack(alignment: .leading, spacing: 12) {
                HStack(spacing: 7) {
                    Circle()
                        .fill(statusColor)
                        .frame(width: 6, height: 6)
                        .shadow(color: statusColor.opacity(0.65), radius: 5)
                    Text(statusText)
                        .font(.system(size: 9, weight: .semibold, design: .monospaced))
                        .tracking(0.8)
                }
                Text("Your prompts and videos stay on this Mac.")
                    .font(.system(size: 10.5))
                    .foregroundStyle(FVTheme.faint)
                    .fixedSize(horizontal: false, vertical: true)
            }
            .padding(16)
            .fvLiquidGlass(cornerRadius: 14, tint: statusColor.opacity(0.08))
            .padding(12)
        }
        .foregroundStyle(FVTheme.text)
        .background(PlatformSidebarMaterial())
    }

    private var statusColor: Color {
        switch model.runtimeHealth.state {
        case .ready: FVTheme.lime
        case .checking: FVTheme.amber
        case .needsSetup, .error: FVTheme.red
        }
    }

    private var statusText: String {
        switch model.runtimeHealth.state {
        case .ready: "METAL READY"
        case .checking: "CHECKING"
        case .needsSetup: "SETUP NEEDED"
        case .error: "CHECK FAILED"
        }
    }
}

private struct CreateView: View {
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 0) {
                HeroView()
                ComposerView()
                    .padding(.horizontal, 34)
                    .padding(.top, 18)
                    .padding(.bottom, 54)
            }
        }
        .scrollIndicators(.hidden)
        .background(FVTheme.background)
    }
}

private struct HeroView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ZStack(alignment: .bottomLeading) {
            ASCIIFieldView()
                .opacity(0.62)
                .mask(
                    LinearGradient(
                        colors: [.clear, .white.opacity(0.75), .white, .clear],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
            LinearGradient(
                colors: [.clear, FVTheme.background.opacity(0.32), FVTheme.background],
                startPoint: .top,
                endPoint: .bottom
            )
            VStack(alignment: .leading, spacing: 0) {
                HStack(spacing: 9) {
                    Circle().fill(FVTheme.lime).frame(width: 6, height: 6)
                    Text("LOCAL TEXT-TO-VIDEO / APPLE SILICON")
                        .font(.system(size: 10, weight: .semibold, design: .monospaced))
                        .tracking(1.4)
                        .foregroundStyle(FVTheme.muted)
                }
                .padding(.bottom, 18)

                Text("Five seconds of motion.\nNo cloud in the middle.")
                    .font(.system(size: 52, weight: .medium, design: .rounded))
                    .tracking(-2.3)
                    .foregroundStyle(FVTheme.text)
                    .lineSpacing(-3)
                Text("FastWan-QAD 1.3B runs through MLX and Metal on your Mac. The first rough cut appears while the model is still refining it.")
                    .font(.system(size: 14.5, weight: .regular))
                    .foregroundStyle(FVTheme.muted)
                    .frame(maxWidth: 660, alignment: .leading)
                    .padding(.top, 18)

                HStack(spacing: 28) {
                    HeroStat(value: "3", label: "DMD STEPS")
                    HeroStat(value: "~40s", label: "FIRST LIGHT")
                    HeroStat(value: "0", label: "UPLOADS")
                    HeroStat(value: "1.3B", label: "PARAMETERS")
                }
                .padding(.top, 25)
            }
            .padding(.horizontal, 42)
            .padding(.bottom, 34)
        }
        .frame(height: 432)
        .overlay(alignment: .topTrailing) {
            Text("FASTWAN / QAD / INT8 / \(model.generationSettings.variant.label)")
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .tracking(1.2)
                .foregroundStyle(FVTheme.faint)
                .padding(.top, 36)
                .padding(.trailing, 38)
        }
    }
}

private struct HeroStat: View {
    let value: String
    let label: String

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(value)
                .font(.system(size: 19, weight: .medium, design: .monospaced))
                .foregroundStyle(FVTheme.text)
            Text(label)
                .font(.system(size: 8.5, weight: .semibold, design: .monospaced))
                .tracking(1.1)
                .foregroundStyle(FVTheme.faint)
        }
    }
}

struct ASCIIFieldView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.periodic(from: .now, by: reduceMotion ? 2 : 0.14)) { timeline in
            GeometryReader { proxy in
                Text(field(for: timeline.date.timeIntervalSinceReferenceDate, size: proxy.size))
                    .font(.system(size: 8, weight: .regular, design: .monospaced))
                    .lineSpacing(0.8)
                    .foregroundStyle(FVTheme.lime.opacity(0.52))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .clipped()
            }
        }
        .accessibilityHidden(true)
    }

    private func field(for time: TimeInterval, size: CGSize) -> String {
        let width = max(48, min(130, Int(size.width / 7.3)))
        let height = max(20, min(48, Int(size.height / 9.0)))
        let glyphs = Array("  ..::--==++**##%@")
        let t = reduceMotion ? 0.3 : time * 0.34
        var rows: [String] = []
        rows.reserveCapacity(height)
        for row in 0..<height {
            var characters: [Character] = []
            characters.reserveCapacity(width)
            for column in 0..<width {
                let x = Double(column) / Double(width) * 6.4 - 3.2
                let y = Double(row) / Double(height) * 3.0 - 1.5
                let wave = sin(x * 1.7 + t) + cos(y * 3.1 - t * 0.72)
                let dx = x - sin(t)
                let dy = y - cos(t * 0.8)
                let orbit = sin(sqrt(dx * dx + dy * dy) * 4.4 - t * 2)
                let vignette = max(0, 1 - (x * x / 13 + y * y / 3.2))
                let value = max(0, min(0.999, (wave + orbit + 3) / 6 * vignette))
                characters.append(glyphs[Int(value * Double(glyphs.count))])
            }
            rows.append(String(characters))
        }
        return rows.joined(separator: "\n")
    }
}

private struct ComposerView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if !model.runtimeHealth.canGenerate {
                SetupBanner()
            }
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 14) {
                    PromptPanel().frame(minWidth: 380)
                    OutputPanel(record: model.activeRecord).frame(minWidth: 410)
                }
                VStack(spacing: 14) {
                    PromptPanel()
                    OutputPanel(record: model.activeRecord)
                }
            }
        }
    }
}

private struct SetupBanner: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        HStack(spacing: 13) {
            Image(systemName: "wrench.and.screwdriver")
                .foregroundStyle(FVTheme.amber)
            VStack(alignment: .leading, spacing: 2) {
                Text("One-time setup is not finished")
                    .font(.system(size: 12.5, weight: .semibold))
                Text("Connect the MLX runtime, ffmpeg, and at least one model checkpoint.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(FVTheme.muted)
            }
            Spacer()
            Button("Open setup") { model.section = .setup }
                .buttonStyle(FVSecondaryButtonStyle())
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 9).fill(FVTheme.amber.opacity(0.075)))
        .overlay(RoundedRectangle(cornerRadius: 9).stroke(FVTheme.amber.opacity(0.25)))
    }
}

private struct PromptPanel: View {
    @EnvironmentObject private var model: AppModel
    @State private var showAdvanced = false

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            PanelHeader(index: "01", title: "Direct the shot", trailing: "TEXT → VIDEO")
            ZStack(alignment: .topLeading) {
                TextEditor(text: $model.prompt)
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .scrollContentBackground(.hidden)
                    .padding(12)
                    .frame(minHeight: 142)
                    .background(Color.black.opacity(0.18))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(FVTheme.line))
                if model.prompt.isEmpty {
                    Text("Describe the subject, motion, camera, and light…")
                        .font(.system(size: 14))
                        .foregroundStyle(FVTheme.faint)
                        .padding(18)
                        .allowsHitTesting(false)
                }
            }

            VStack(alignment: .leading, spacing: 10) {
                SectionLabel("WEIGHTS")
                HStack(spacing: 8) {
                    ForEach(ModelVariant.allCases) { variant in
                        VariantButton(
                            variant: variant,
                            selected: model.generationSettings.variant == variant,
                            available: model.runtimeHealth.variantAvailable(variant)
                        ) {
                            model.generationSettings.variant = variant
                        }
                    }
                }
            }

            HStack(spacing: 9) {
                SettingMenu(
                    label: "FRAME",
                    value: model.generationSettings.resolutionLabel,
                    options: [("832 × 480", (832, 480)), ("672 × 384", (672, 384)), ("448 × 256", (448, 256))]
                ) { size in
                    model.generationSettings.width = size.0
                    model.generationSettings.height = size.1
                }
                SettingMenu(
                    label: "LENGTH",
                    value: String(format: "%.1f sec", model.generationSettings.duration),
                    options: [("2.1 sec", 33), ("3.1 sec", 49), ("5.1 sec", 81)]
                ) { model.generationSettings.frames = $0 }
                SettingMenu(
                    label: "FPS",
                    value: "\(model.generationSettings.fps)",
                    options: [("16 fps", 16), ("24 fps", 24)]
                ) { model.generationSettings.fps = $0 }
            }

            DisclosureGroup(isExpanded: $showAdvanced) {
                VStack(alignment: .leading, spacing: 13) {
                    HStack {
                        Text("Seed")
                        Spacer()
                        TextField("Seed", value: $model.generationSettings.seed, format: .number)
                            .textFieldStyle(.plain)
                            .multilineTextAlignment(.trailing)
                            .font(.system(size: 11, design: .monospaced))
                            .frame(width: 90)
                    }
                    Toggle("Parallel TAEHV decode", isOn: $model.generationSettings.parallelDecode)
                    Toggle(
                        "Cap MLX memory",
                        isOn: Binding(
                            get: { model.generationSettings.memoryLimitGiB != nil },
                            set: { model.generationSettings.memoryLimitGiB = $0 ? 16 : nil }
                        )
                    )
                    if model.generationSettings.memoryLimitGiB != nil {
                        HStack {
                            Slider(
                                value: Binding(
                                    get: { model.generationSettings.memoryLimitGiB ?? 16 },
                                    set: { model.generationSettings.memoryLimitGiB = $0 }
                                ),
                                in: 8...48,
                                step: 1
                            )
                            Text("\(Int(model.generationSettings.memoryLimitGiB ?? 16)) GiB")
                                .font(.system(size: 10, design: .monospaced))
                                .frame(width: 46)
                        }
                    }
                }
                .font(.system(size: 11.5))
                .foregroundStyle(FVTheme.muted)
                .padding(.top, 12)
            } label: {
                Text("ADVANCED CONTROLS")
                    .font(.system(size: 9, weight: .semibold, design: .monospaced))
                    .tracking(1)
                    .foregroundStyle(FVTheme.faint)
            }

            HStack(spacing: 12) {
                Button {
                    model.isGenerating ? model.cancelGeneration() : model.generate()
                } label: {
                    HStack {
                        Image(systemName: model.isGenerating ? "stop.fill" : "play.fill")
                        Text(model.isGenerating ? "Stop generation" : "Generate on this Mac")
                        Spacer()
                        Text(model.isGenerating ? "" : "⌘↩")
                            .font(.system(size: 10, design: .monospaced))
                            .opacity(0.55)
                    }
                }
                .buttonStyle(FVPrimaryButtonStyle(danger: model.isGenerating))
                .disabled(!model.isGenerating && !model.runtimeHealth.canGenerate)
            }
        }
        .panelStyle()
    }
}

private struct VariantButton: View {
    let variant: ModelVariant
    let selected: Bool
    let available: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    Text(variant.label)
                        .font(.system(size: 11, weight: .bold, design: .monospaced))
                    Text(variant.detail)
                        .font(.system(size: 9.5))
                        .foregroundStyle(selected ? Color.black.opacity(0.62) : FVTheme.faint)
                }
                Spacer()
                Circle()
                    .fill(available ? (selected ? Color.black.opacity(0.72) : FVTheme.lime) : FVTheme.faint)
                    .frame(width: 6, height: 6)
            }
            .padding(.horizontal, 12)
            .frame(maxWidth: .infinity, minHeight: 50)
            .foregroundStyle(selected ? Color.black : FVTheme.text)
            .background(RoundedRectangle(cornerRadius: 7).fill(selected ? FVTheme.lime : Color.black.opacity(0.16)))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(selected ? .clear : FVTheme.line))
        }
        .buttonStyle(.plain)
        .disabled(!available)
        .opacity(available ? 1 : 0.44)
    }
}

private struct SettingMenu<Value>: View {
    let label: String
    let value: String
    let options: [(String, Value)]
    let select: (Value) -> Void

    var body: some View {
        Menu {
            ForEach(Array(options.enumerated()), id: \.offset) { _, option in
                Button(option.0) { select(option.1) }
            }
        } label: {
            VStack(alignment: .leading, spacing: 5) {
                Text(label)
                    .font(.system(size: 8, weight: .semibold, design: .monospaced))
                    .tracking(0.9)
                    .foregroundStyle(FVTheme.faint)
                HStack(spacing: 5) {
                    Text(value)
                        .font(.system(size: 11, weight: .medium, design: .monospaced))
                    Image(systemName: "chevron.down")
                        .font(.system(size: 7, weight: .bold))
                        .foregroundStyle(FVTheme.faint)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 11)
            .frame(height: 50)
            .background(RoundedRectangle(cornerRadius: 7).fill(Color.black.opacity(0.15)))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(FVTheme.line))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
    }
}

private struct OutputPanel: View {
    @EnvironmentObject private var model: AppModel
    let record: GenerationRecord?

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            PanelHeader(index: "02", title: "Watch it emerge", trailing: record?.status.rawValue.uppercased() ?? "WAITING")
            ZStack {
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.black.opacity(0.36))
                if let url = record?.playbackURL,
                   FileManager.default.fileExists(atPath: url.path) {
                    VideoSurface(url: url)
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .id(url.path)
                    if record?.status == .running {
                        VStack {
                            HStack {
                                LiveBadge()
                                Spacer()
                            }
                            Spacer()
                        }
                        .padding(12)
                    }
                } else if let record, record.status == .running {
                    GeneratingPlaceholder(record: record)
                } else {
                    EmptyOutputView()
                }
            }
            .aspectRatio(832.0 / 480.0, contentMode: .fit)
            .overlay(RoundedRectangle(cornerRadius: 8).stroke(FVTheme.line))

            if let record {
                HStack(alignment: .center, spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(record.phase.uppercased())
                            .font(.system(size: 9, weight: .bold, design: .monospaced))
                            .tracking(1)
                            .foregroundStyle(record.status == .failed ? FVTheme.red : FVTheme.muted)
                        if record.status == .running {
                            ProgressView(value: record.progress)
                                .progressViewStyle(.linear)
                                .tint(FVTheme.lime)
                                .frame(maxWidth: 260)
                        } else if let metrics = record.metrics, let total = metrics.totalSeconds {
                            Text("Finished in \(total.formatted(.number.precision(.fractionLength(1)))) seconds")
                                .font(.system(size: 10.5))
                                .foregroundStyle(FVTheme.faint)
                        }
                    }
                    Spacer()
                    if record.status == .completed, let url = record.outputURL {
                        HStack(spacing: 7) {
                            Button { model.reveal(record) } label: { Image(systemName: "folder") }
                                .buttonStyle(FVIconButtonStyle())
                                .help("Reveal in Finder")
                            Button { model.export(record) } label: { Image(systemName: "arrow.down.to.line") }
                                .buttonStyle(FVIconButtonStyle())
                                .help("Export a copy")
                            ShareLink(item: url) { Image(systemName: "square.and.arrow.up") }
                                .buttonStyle(FVIconButtonStyle())
                                .help("Share")
                        }
                    }
                }
            } else {
                Text("The first x0 preview appears after DMD step 1. The final video replaces it automatically.")
                    .font(.system(size: 10.5))
                    .foregroundStyle(FVTheme.faint)
            }
        }
        .panelStyle()
    }
}

private struct VideoSurface: NSViewRepresentable {
    let url: URL

    final class Coordinator {
        var url: URL?
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        updatePlayer(in: view, context: context)
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) {
        updatePlayer(in: view, context: context)
    }

    static func dismantleNSView(_ view: AVPlayerView, coordinator: Coordinator) {
        view.player?.pause()
        view.player = nil
        coordinator.url = nil
    }

    private func updatePlayer(in view: AVPlayerView, context: Context) {
        guard context.coordinator.url != url else { return }
        view.player?.pause()
        let player = AVPlayer(url: url)
        player.isMuted = true
        view.player = player
        context.coordinator.url = url
        player.play()
    }
}

private struct LiveBadge: View {
    var body: some View {
        HStack(spacing: 6) {
            Circle().fill(FVTheme.lime).frame(width: 5, height: 5)
            Text("LIVE X0 PREVIEW")
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .tracking(0.8)
        }
        .foregroundStyle(Color.black.opacity(0.82))
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .background(Capsule().fill(FVTheme.lime))
    }
}

private struct GeneratingPlaceholder: View {
    let record: GenerationRecord

    var body: some View {
        VStack(spacing: 17) {
            ZStack {
                Circle().stroke(FVTheme.line, lineWidth: 3)
                Circle()
                    .trim(from: 0, to: max(0.03, record.progress))
                    .stroke(FVTheme.lime, style: StrokeStyle(lineWidth: 3, lineCap: .round))
                    .rotationEffect(.degrees(-90))
                Text("\(Int(record.progress * 100))")
                    .font(.system(size: 17, weight: .medium, design: .monospaced))
            }
            .frame(width: 64, height: 64)
            VStack(spacing: 5) {
                Text(record.phase)
                    .font(.system(size: 12, weight: .semibold))
                Text("The first rough cut will begin playing here.")
                    .font(.system(size: 10.5))
                    .foregroundStyle(FVTheme.faint)
            }
        }
    }
}

private struct EmptyOutputView: View {
    var body: some View {
        VStack(spacing: 13) {
            Text("┌─────────────────┐\n│  YOUR NEXT SHOT │\n│     · · ·       │\n└─────────────────┘")
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(FVTheme.faint)
                .multilineTextAlignment(.center)
            Text("Ready when you are")
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(FVTheme.muted)
        }
    }
}

private struct LibraryView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PageHeader(
                eyebrow: "LOCAL ARCHIVE",
                title: "Every generation, still yours.",
                detail: "Outputs, settings, timing, and interrupted attempts live here on this Mac."
            )
            if model.records.isEmpty {
                Spacer()
                VStack(spacing: 14) {
                    Image(systemName: "film.stack")
                        .font(.system(size: 34, weight: .thin))
                        .foregroundStyle(FVTheme.faint)
                    Text("No generations yet")
                        .font(.system(size: 16, weight: .semibold))
                    Button("Create the first one") { model.section = .create }
                        .buttonStyle(FVSecondaryButtonStyle())
                }
                .frame(maxWidth: .infinity)
                Spacer()
            } else {
                HStack(alignment: .top, spacing: 14) {
                    ScrollView {
                        LazyVStack(spacing: 8) {
                            ForEach(model.records) { record in
                                LibraryRow(record: record, selected: model.selectedRecordID == record.id)
                                    .onTapGesture { model.selectedRecordID = record.id }
                            }
                        }
                    }
                    .frame(width: 320)
                    OutputPanel(record: model.activeRecord)
                        .frame(maxWidth: .infinity)
                }
                .padding(.horizontal, 34)
                .padding(.bottom, 34)
            }
        }
        .foregroundStyle(FVTheme.text)
        .background(FVTheme.background)
    }
}

private struct LibraryRow: View {
    @EnvironmentObject private var model: AppModel
    let record: GenerationRecord
    let selected: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(record.settings.variant.label)
                    .font(.system(size: 8.5, weight: .bold, design: .monospaced))
                    .foregroundStyle(statusColor)
                Text(record.createdAt.formatted(date: .abbreviated, time: .shortened))
                    .font(.system(size: 9.5, design: .monospaced))
                    .foregroundStyle(FVTheme.faint)
                Spacer()
                Circle().fill(statusColor).frame(width: 6, height: 6)
            }
            Text(record.prompt)
                .font(.system(size: 12.5, weight: .medium))
                .lineLimit(3)
                .frame(maxWidth: .infinity, alignment: .leading)
            HStack {
                Text("\(record.settings.resolutionLabel) · \(record.settings.frames)f")
                    .font(.system(size: 9, design: .monospaced))
                    .foregroundStyle(FVTheme.faint)
                Spacer()
                if selected, record.status != .running {
                    Button(role: .destructive) { model.delete(record) } label: {
                        Image(systemName: "trash")
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(FVTheme.faint)
                }
            }
        }
        .padding(14)
        .background(RoundedRectangle(cornerRadius: 9).fill(selected ? FVTheme.surfaceStrong : FVTheme.surface))
        .overlay(RoundedRectangle(cornerRadius: 9).stroke(selected ? FVTheme.lime.opacity(0.42) : FVTheme.line))
        .contentShape(Rectangle())
    }

    private var statusColor: Color {
        switch record.status {
        case .completed: FVTheme.lime
        case .running, .queued: FVTheme.amber
        case .failed, .cancelled: FVTheme.red
        }
    }
}

private struct SetupView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("fastvideo.onboarding.completed") private var onboardingCompleted = true

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                PageHeader(
                    eyebrow: "ONE-TIME SETUP",
                    title: "Make this Mac a video machine.",
                    detail: "The app manages a local Python environment and model folder. Nothing here starts a cloud service."
                )
                if case let .error(message) = model.runtimeHealth.state {
                    HStack(alignment: .top, spacing: 10) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(FVTheme.red)
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Runtime check failed")
                                .font(.system(size: 12, weight: .semibold))
                            Text(message)
                                .font(.system(size: 10, design: .monospaced))
                                .foregroundStyle(FVTheme.muted)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(13)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(FVTheme.red.opacity(0.08)))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(FVTheme.red.opacity(0.25)))
                }
                HStack(alignment: .top, spacing: 12) {
                    SetupCard(
                        index: "01",
                        title: "Apple runtime",
                        detail: "Python 3.12, FastVideo, MLX 0.31, PyTorch MPS",
                        ready: model.runtimeHealth.mlxAvailable && model.runtimeHealth.mpsAvailable,
                        actionTitle: model.isInstallingRuntime ? "Installing…" : "Install runtime",
                        action: model.installRuntime
                    )
                    SetupCard(
                        index: "02",
                        title: "Video tools",
                        detail: "ffmpeg for the final local MP4 export",
                        ready: model.runtimeHealth.ffmpegAvailable,
                        actionTitle: "Install ffmpeg",
                        action: model.installFFmpeg
                    )
                    SetupCard(
                        index: "03",
                        title: "FastWan-QAD v2 1.3B",
                        detail: "Shared model files plus RAW and/or EMA MLX checkpoints",
                        ready: model.runtimeHealth.modelComponentsPresent && (model.runtimeHealth.rawAvailable || model.runtimeHealth.emaAvailable),
                        actionTitle: model.isInstallingModel ? "Downloading…" : "Download model",
                        action: model.installModel
                    )
                }

                if model.isInstallingModel {
                    VStack(alignment: .leading, spacing: 7) {
                        HStack {
                            Text("MODEL DOWNLOAD")
                                .font(.system(size: 9, weight: .bold, design: .monospaced))
                                .tracking(1)
                            Spacer()
                            if let progress = model.modelInstallProgress {
                                Text("\(Int(progress * 100))%")
                                    .font(.system(size: 9, design: .monospaced))
                            }
                        }
                        ProgressView(value: model.modelInstallProgress)
                            .tint(FVTheme.lime)
                    }
                    .padding(14)
                    .background(RoundedRectangle(cornerRadius: 8).fill(FVTheme.surface))
                }

                VStack(alignment: .leading, spacing: 14) {
                    PanelHeader(index: "04", title: "Paths and release source", trailing: "ADVANCED")
                    PathRow(label: "FASTVIDEO SOURCE", value: $model.configuration.repositoryRoot, choose: model.chooseRepositoryRoot)
                    PathRow(label: "PYTHON", value: $model.configuration.pythonExecutable, choose: model.choosePython)
                    PathRow(label: "MODEL FOLDER", value: $model.configuration.modelRoot, choose: model.chooseModelRoot)
                    VStack(alignment: .leading, spacing: 6) {
                        SectionLabel("HUGGING FACE REPOSITORY")
                        TextField("Repository", text: $model.configuration.modelRepository)
                            .textFieldStyle(FVTextFieldStyle())
                    }
                    HStack(spacing: 10) {
                        VStack(alignment: .leading, spacing: 6) {
                            SectionLabel("RAW CHECKPOINT OVERRIDE")
                            TextField("Auto-detect", text: $model.configuration.rawCheckpoint)
                                .textFieldStyle(FVTextFieldStyle())
                        }
                        VStack(alignment: .leading, spacing: 6) {
                            SectionLabel("EMA CHECKPOINT OVERRIDE")
                            TextField("Auto-detect", text: $model.configuration.emaCheckpoint)
                                .textFieldStyle(FVTextFieldStyle())
                        }
                    }
                    HStack {
                        Text("EMA is the release default. RAW remains available for direct A/B comparison when its local checkpoint is connected.")
                            .font(.system(size: 10.5))
                            .foregroundStyle(FVTheme.faint)
                        Spacer()
                        Button("Replay welcome") { onboardingCompleted = false }
                            .buttonStyle(FVSecondaryButtonStyle())
                        Button("Save and re-check") { model.saveConfiguration() }
                            .buttonStyle(FVSecondaryButtonStyle())
                    }
                }
                .panelStyle()

                if !model.setupLog.isEmpty {
                    VStack(alignment: .leading, spacing: 9) {
                        SectionLabel("SETUP LOG")
                        ScrollView {
                            Text(model.setupLog.suffix(120).joined(separator: "\n"))
                                .font(.system(size: 9.5, design: .monospaced))
                                .foregroundStyle(FVTheme.muted)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .textSelection(.enabled)
                        }
                        .frame(maxHeight: 180)
                    }
                    .padding(14)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.black.opacity(0.28)))
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(FVTheme.line))
                }
            }
            .padding(.horizontal, 34)
            .padding(.bottom, 44)
        }
        .foregroundStyle(FVTheme.text)
        .background(FVTheme.background)
    }
}

private struct SetupCard: View {
    let index: String
    let title: String
    let detail: String
    let ready: Bool
    let actionTitle: String
    let action: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text(index)
                    .font(.system(size: 9, weight: .bold, design: .monospaced))
                    .foregroundStyle(FVTheme.faint)
                Spacer()
                HStack(spacing: 5) {
                    Circle().fill(ready ? FVTheme.lime : FVTheme.amber).frame(width: 6, height: 6)
                    Text(ready ? "READY" : "NEEDED")
                        .font(.system(size: 8, weight: .bold, design: .monospaced))
                }
                .foregroundStyle(ready ? FVTheme.lime : FVTheme.amber)
            }
            Text(title)
                .font(.system(size: 15, weight: .semibold))
            Text(detail)
                .font(.system(size: 10.5))
                .foregroundStyle(FVTheme.muted)
                .frame(maxWidth: .infinity, minHeight: 32, alignment: .topLeading)
            Button(ready ? "Reinstall" : actionTitle, action: action)
                .buttonStyle(FVSecondaryButtonStyle())
                .disabled(actionTitle.hasSuffix("…"))
        }
        .padding(15)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 9).fill(FVTheme.surface))
        .overlay(RoundedRectangle(cornerRadius: 9).stroke(ready ? FVTheme.lime.opacity(0.22) : FVTheme.line))
    }
}

private struct PathRow: View {
    let label: String
    @Binding var value: String
    let choose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            SectionLabel(label)
            HStack(spacing: 8) {
                TextField(label, text: $value)
                    .textFieldStyle(FVTextFieldStyle())
                Button("Choose…", action: choose)
                    .buttonStyle(FVSecondaryButtonStyle())
            }
        }
    }
}

private struct PageHeader: View {
    let eyebrow: String
    let title: String
    let detail: String

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            Text(eyebrow)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .tracking(1.3)
                .foregroundStyle(FVTheme.lime)
            Text(title)
                .font(.system(size: 31, weight: .medium, design: .rounded))
                .tracking(-1)
            Text(detail)
                .font(.system(size: 12.5))
                .foregroundStyle(FVTheme.muted)
        }
        .padding(.top, 46)
        .padding(.bottom, 26)
    }
}

private struct PanelHeader: View {
    let index: String
    let title: String
    let trailing: String

    var body: some View {
        HStack(spacing: 10) {
            Text(index)
                .font(.system(size: 9, weight: .bold, design: .monospaced))
                .foregroundStyle(FVTheme.lime)
            Text(title)
                .font(.system(size: 14, weight: .semibold))
            Spacer()
            Text(trailing)
                .font(.system(size: 8.5, weight: .semibold, design: .monospaced))
                .tracking(0.9)
                .foregroundStyle(FVTheme.faint)
        }
    }
}

private struct SectionLabel: View {
    let value: String
    init(_ value: String) { self.value = value }
    var body: some View {
        Text(value)
            .font(.system(size: 8.5, weight: .semibold, design: .monospaced))
            .tracking(1)
            .foregroundStyle(FVTheme.faint)
    }
}

private struct FVPrimaryButtonStyle: ButtonStyle {
    var danger = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 12.5, weight: .semibold))
            .foregroundStyle(danger ? FVTheme.text : Color.black.opacity(0.88))
            .padding(.horizontal, 16)
            .frame(maxWidth: .infinity, minHeight: 44)
            .background(
                RoundedRectangle(cornerRadius: 8)
                    .fill(danger ? FVTheme.red.opacity(configuration.isPressed ? 0.58 : 0.74) : FVTheme.lime.opacity(configuration.isPressed ? 0.72 : 1))
            )
            .opacity(configuration.isPressed ? 0.82 : 1)
    }
}

private struct FVSecondaryButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 10.5, weight: .semibold))
            .foregroundStyle(FVTheme.muted)
            .padding(.horizontal, 12)
            .frame(height: 30)
            .background(RoundedRectangle(cornerRadius: 6).fill(configuration.isPressed ? FVTheme.surfaceStrong : FVTheme.surface))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(FVTheme.line))
    }
}

private struct FVIconButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(FVTheme.muted)
            .frame(width: 30, height: 30)
            .background(RoundedRectangle(cornerRadius: 6).fill(configuration.isPressed ? FVTheme.surfaceStrong : FVTheme.surface))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(FVTheme.line))
    }
}

private struct FVTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .font(.system(size: 10.5, design: .monospaced))
            .padding(.horizontal, 11)
            .frame(height: 34)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color.black.opacity(0.18)))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(FVTheme.line))
    }
}

private extension View {
    func panelStyle() -> some View {
        padding(18)
            .background(RoundedRectangle(cornerRadius: 11).fill(FVTheme.surface))
            .overlay(RoundedRectangle(cornerRadius: 11).stroke(FVTheme.line))
    }
}
