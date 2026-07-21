import AVKit
import SwiftUI

enum FVTheme {
    static let background = Color(red: 0.025, green: 0.028, blue: 0.026)
    static let sidebar = Color(red: 0.045, green: 0.049, blue: 0.046)
    static let surface = Color.white.opacity(0.052)
    static let surfaceStrong = Color.white.opacity(0.085)
    static let line = Color.white.opacity(0.11)
    static let text = Color.white.opacity(0.94)
    static let muted = Color.white.opacity(0.58)
    static let faint = Color.white.opacity(0.32)
    static let lime = Color(red: 0.69, green: 0.96, blue: 0.42)
    static let mint = Color(red: 0.42, green: 0.87, blue: 0.68)
    static let amber = Color(red: 0.96, green: 0.70, blue: 0.32)
    static let red = Color(red: 0.96, green: 0.36, blue: 0.32)

    // Ambient aurora tones. Deliberately warm/cool natural light — no purple.
    static let glowTeal = Color(red: 0.24, green: 0.72, blue: 0.64)
    static let glowWarm = Color(red: 0.91, green: 0.66, blue: 0.36)

    /// Brand gradient: lime easing into mint. Used for marks and key accents.
    static var accentGradient: LinearGradient {
        LinearGradient(
            colors: [lime, mint],
            startPoint: .topLeading,
            endPoint: .bottomTrailing
        )
    }

    /// Quiet vertical sheen for hero titles.
    static var heroGradient: LinearGradient {
        LinearGradient(
            colors: [
                Color.white.opacity(0.98),
                Color(red: 0.80, green: 0.93, blue: 0.82).opacity(0.72),
            ],
            startPoint: .top,
            endPoint: .bottom
        )
    }
}

struct RootView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("fastvideo.onboarding.completed") private var onboardingCompleted = false

    var body: some View {
        Group {
            if onboardingCompleted {
                AppShellView()
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
            "FastWan QAD",
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

private struct AppShellView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationSplitView {
            SidebarView()
                .navigationSplitViewColumnWidth(min: 188, ideal: 212, max: 236)
        } detail: {
            Group {
                switch model.section {
                case .create: CreateView()
                case .library: LibraryView()
                case .setup: SetupView()
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .background(FVTheme.background)
        }
        .navigationSplitViewStyle(.balanced)
    }
}

private struct SidebarView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 11) {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(FVTheme.accentGradient)
                    .frame(width: 32, height: 32)
                    .overlay {
                        Text("FQ")
                            .font(.system(size: 10, weight: .black, design: .rounded))
                            .foregroundStyle(Color.black.opacity(0.82))
                    }
                    .shadow(color: FVTheme.lime.opacity(0.22), radius: 10, y: 3)
                VStack(alignment: .leading, spacing: 1) {
                    Text("FastWan QAD")
                        .font(.system(size: 13, weight: .semibold, design: .rounded))
                    Text("On-device video")
                        .font(.system(size: 10))
                        .foregroundStyle(FVTheme.faint)
                }
                .lineLimit(1)
            }
            .padding(.horizontal, 17)
            .padding(.top, 50)
            .padding(.bottom, 28)

            VStack(spacing: 5) {
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
                            Spacer(minLength: 8)
                            if section == .library, !model.records.isEmpty {
                                Text("\(model.records.count)")
                                    .font(.system(size: 10, weight: .medium, design: .rounded))
                                    .foregroundStyle(FVTheme.faint)
                            }
                        }
                        .foregroundStyle(model.section == section ? FVTheme.text : FVTheme.muted)
                        .padding(.horizontal, 11)
                        .frame(height: 40)
                        .background {
                            if model.section == section {
                                if #available(macOS 26.0, *) {
                                    Color.clear
                                        .glassEffect(.regular.tint(Color.white.opacity(0.10)), in: .rect(cornerRadius: 9))
                                } else {
                                    RoundedRectangle(cornerRadius: 9, style: .continuous)
                                        .fill(Color.white.opacity(0.085))
                                }
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 9)

            Spacer(minLength: 16)

            HStack(spacing: 9) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 7, height: 7)
                    .shadow(color: statusColor.opacity(0.6), radius: 4)
                VStack(alignment: .leading, spacing: 2) {
                    Text(statusTitle)
                        .font(.system(size: 11, weight: .medium))
                    Text(statusDetail)
                        .font(.system(size: 9.5))
                        .foregroundStyle(FVTheme.faint)
                        .lineLimit(1)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .fvLiquidGlass(cornerRadius: 14, tint: statusColor.opacity(0.05))
            .padding(.horizontal, 12)
            .padding(.bottom, 14)
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

    private var statusTitle: String {
        switch model.runtimeHealth.state {
        case .ready: "Ready on this Mac"
        case .checking: "Checking this Mac"
        case .needsSetup: "Setup required"
        case .error: "Check failed"
        }
    }

    private var statusDetail: String {
        switch model.runtimeHealth.state {
        case .ready: "MLX and Metal available"
        case .checking: "Verifying local runtime"
        case .needsSetup: "Open Models & Runtime"
        case .error: "Review runtime details"
        }
    }
}

private struct CreateView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        ZStack {
            FVAmbientBackground(intensity: model.createRecord == nil ? 1 : 0.5)
                .ignoresSafeArea()
            ASCIIFieldView()
                .opacity(model.createRecord == nil ? 0.30 : 0.11)
                .mask(
                    RadialGradient(
                        colors: [.white, .white.opacity(0.7), .clear],
                        center: model.createRecord == nil ? .center : .topTrailing,
                        startRadius: 20,
                        endRadius: 700
                    )
                )
                .ignoresSafeArea()
            if let record = model.createRecord {
                GenerationWorkspace(record: record)
                    .transition(.opacity.combined(with: .scale(scale: 0.985)))
            } else {
                PromptHome()
                    .transition(.opacity)
            }
        }
        .animation(.easeInOut(duration: 0.32), value: model.createRecord?.id)
    }
}

private struct PromptHome: View {
    @EnvironmentObject private var model: AppModel
    @FocusState private var promptFocused: Bool

    var body: some View {
        VStack(spacing: 0) {
            Spacer(minLength: 80)
            VStack(spacing: 12) {
                Text("FastWan QAD")
                    .font(.system(size: 58, weight: .medium, design: .rounded))
                    .tracking(-2.6)
                    .foregroundStyle(FVTheme.heroGradient)
                Text("Create video locally on Apple silicon.")
                    .font(.system(size: 15, weight: .regular, design: .rounded))
                    .foregroundStyle(FVTheme.muted)
            }
            .padding(.bottom, 32)

            HStack(alignment: .bottom, spacing: 10) {
                TextField("Describe a scene and how it moves…", text: $model.prompt, axis: .vertical)
                    .textFieldStyle(.plain)
                    .font(.system(size: 16, weight: .regular, design: .rounded))
                    .lineLimit(1...5)
                    .focused($promptFocused)
                    .padding(.leading, 6)
                    .padding(.vertical, 10)

                GenerationOptionsMenu()

                Button {
                    model.generate()
                } label: {
                    if #available(macOS 26.0, *) {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundStyle(Color.black.opacity(0.82))
                            .frame(width: 38, height: 38)
                            .glassEffect(.regular.tint(FVTheme.lime).interactive(true), in: .circle)
                    } else {
                        Image(systemName: "arrow.up")
                            .font(.system(size: 15, weight: .bold))
                            .foregroundStyle(Color.black.opacity(0.82))
                            .frame(width: 38, height: 38)
                            .background(Circle().fill(FVTheme.accentGradient))
                    }
                }
                .buttonStyle(.plain)
                .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .opacity(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.42 : 1)
                .help("Generate video")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(maxWidth: 760, minHeight: 64)
            .fvLiquidGlass(cornerRadius: 22, tint: FVTheme.lime.opacity(0.02), interactive: true)
            .shadow(color: Color.black.opacity(0.25), radius: 30, y: 16)

            HStack(spacing: 8) {
                PromptChip(label: "Misty forest", prompt: "A fox runs through a misty pine forest, leaves kicking up behind it.")
                PromptChip(label: "Neon rain", prompt: "A lone tram glides through neon rain at midnight, reflections stretching across wet asphalt.")
                PromptChip(label: "Paper world", prompt: "A paper boat sails through a miniature city made from folded maps, warm afternoon light.")
            }
            .padding(.top, 16)

            HStack(spacing: 7) {
                Text("⌘⏎")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(FVTheme.faint)
                    .padding(.horizontal, 7)
                    .frame(height: 20)
                    .background(RoundedRectangle(cornerRadius: 5, style: .continuous).fill(Color.white.opacity(0.055)))
                    .overlay(RoundedRectangle(cornerRadius: 5, style: .continuous).stroke(Color.white.opacity(0.09), lineWidth: 0.5))
                Text("to generate")
                Text("·")
                Text("Fast mode is about 2.7× faster")
            }
            .font(.system(size: 10.5))
            .foregroundStyle(FVTheme.faint)
            .padding(.top, 14)

            Spacer(minLength: 120)
        }
        .padding(.horizontal, 48)
        .onAppear { promptFocused = true }
    }
}

private struct PromptChip: View {
    @EnvironmentObject private var model: AppModel
    let label: String
    let prompt: String

    var body: some View {
        Button(label) { model.prompt = prompt }
            .buttonStyle(.plain)
            .font(.system(size: 11, weight: .medium))
            .foregroundStyle(FVTheme.muted)
            .padding(.horizontal, 12)
            .frame(height: 30)
            .fvLiquidGlass(cornerRadius: 15, interactive: true)
    }
}

private struct GenerationOptionsMenu: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Menu {
            Section("Generation") {
                ForEach(GenerationMode.allCases) { mode in
                    Button {
                        model.generationSettings.mode = mode
                    } label: {
                        Label(
                            mode == .fast
                                ? (model.runtimeHealth.rifeAvailable
                                    ? "Fast · ~2.7× faster"
                                    : "Fast · Install latest runtime")
                                : "Full · native frames",
                            systemImage: model.generationSettings.mode == mode ? "checkmark" : "circle"
                        )
                    }
                    .disabled(mode == .fast && !model.runtimeHealth.rifeAvailable)
                }
            }
            Section("Model") {
                ForEach(ModelVariant.allCases) { variant in
                    Button {
                        model.generationSettings.variant = variant
                    } label: {
                        Label(
                            variant == .ema ? "EMA (Recommended)" : "RAW",
                            systemImage: model.generationSettings.variant == variant ? "checkmark" : "circle"
                        )
                    }
                    .disabled(!model.runtimeHealth.variantAvailable(variant))
                }
            }
            Section("Format") {
                Button("832 × 480 · 5 seconds") {
                    model.generationSettings.width = 832
                    model.generationSettings.height = 480
                    model.generationSettings.frames = 81
                    model.generationSettings.fps = 16
                }
                Button("448 × 256 · 2 seconds") {
                    model.generationSettings.width = 448
                    model.generationSettings.height = 256
                    model.generationSettings.frames = 33
                    model.generationSettings.fps = 16
                }
            }
        } label: {
            if #available(macOS 26.0, *) {
                Image(systemName: model.generationSettings.mode == .fast ? "bolt.fill" : "slider.horizontal.3")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(model.generationSettings.mode == .fast ? FVTheme.lime : FVTheme.muted)
                    .frame(width: 36, height: 36)
                    .glassEffect(
                        .regular.tint(model.generationSettings.mode == .fast ? FVTheme.lime.opacity(0.35) : nil).interactive(true),
                        in: .circle
                    )
            } else {
                Image(systemName: model.generationSettings.mode == .fast ? "bolt.fill" : "slider.horizontal.3")
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(model.generationSettings.mode == .fast ? FVTheme.lime : FVTheme.muted)
                    .frame(width: 36, height: 36)
                    .background(Circle().fill(Color.white.opacity(0.065)))
            }
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .help(model.generationSettings.mode == .fast ? "Fast mode enabled · about 2.7× faster" : "Generation options")
    }
}

private struct GenerationWorkspace: View {
    @EnvironmentObject private var model: AppModel
    let record: GenerationRecord

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 14) {
                Button {
                    model.startNewGeneration()
                } label: {
                    Label("New video", systemImage: "plus")
                }
                .buttonStyle(FVQuietButtonStyle())

                Text(record.prompt)
                    .font(.system(size: 13, weight: .regular, design: .rounded))
                    .foregroundStyle(FVTheme.muted)
                    .lineLimit(1)
                Spacer()
                StatusPill(record: record)
            }
            .padding(.horizontal, 30)
            .padding(.top, 42)
            .padding(.bottom, 20)

            VStack(spacing: 18) {
                ZStack {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(Color.black.opacity(0.48))
                    if let url = record.playbackURL, FileManager.default.fileExists(atPath: url.path) {
                        VideoSurface(url: url)
                            .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                            .id(url.path)
                    } else if record.status == .failed {
                        FailureView(record: record)
                    } else {
                        GeneratingPlaceholder(record: record)
                    }
                    if record.status == .running, record.previewPath != nil {
                        VStack {
                            HStack {
                                LiveBadge()
                                Spacer()
                            }
                            Spacer()
                        }
                        .padding(16)
                    }
                }
                .aspectRatio(832.0 / 480.0, contentMode: .fit)
                .frame(maxWidth: 1040)
                .background(FVVideoGlow(active: record.status == .running))
                .overlay {
                    if record.status == .running {
                        FVProcessingStroke(cornerRadius: 18)
                    } else {
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(FVTheme.line)
                    }
                }
                .shadow(color: Color.black.opacity(0.34), radius: 34, y: 18)

                GenerationFooter(record: record)
                    .frame(maxWidth: 1040)
            }
            .padding(.horizontal, 30)
            .padding(.bottom, 28)
            Spacer(minLength: 0)
        }
    }
}

/// Soft aurora breathing behind the video frame while it is being created.
private struct FVVideoGlow: View {
    let active: Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var breathe = false

    var body: some View {
        RadialGradient(
            colors: [
                FVTheme.glowTeal.opacity(active ? 0.20 : 0.07),
                FVTheme.lime.opacity(active ? 0.10 : 0.03),
                .clear,
            ],
            center: .center,
            startRadius: 80,
            endRadius: 460
        )
        .padding(-70)
        .blur(radius: 26)
        .scaleEffect(breathe ? 1.1 : 1)
        .onAppear {
            guard !reduceMotion else { return }
            withAnimation(.easeInOut(duration: 3.4).repeatForever(autoreverses: true)) {
                breathe = true
            }
        }
        .allowsHitTesting(false)
    }
}

/// A light beam travelling around the frame while generation is in flight.
private struct FVProcessingStroke: View {
    let cornerRadius: CGFloat
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var spin = false

    var body: some View {
        RoundedRectangle(cornerRadius: cornerRadius, style: .continuous)
            .stroke(
                AngularGradient(
                    colors: [
                        FVTheme.lime.opacity(0),
                        FVTheme.lime.opacity(0.75),
                        FVTheme.mint.opacity(0.45),
                        FVTheme.lime.opacity(0),
                    ],
                    center: .center,
                    angle: .degrees(spin ? 360 : 0)
                ),
                lineWidth: 1.6
            )
            .onAppear {
                guard !reduceMotion else { return }
                withAnimation(.linear(duration: 5).repeatForever(autoreverses: false)) {
                    spin = true
                }
            }
            .allowsHitTesting(false)
    }
}

private struct GenerationFooter: View {
    @EnvironmentObject private var model: AppModel
    let record: GenerationRecord

    var body: some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 7) {
                Text(footerTitle)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                if record.status == .running {
                    ProgressView(value: record.progress)
                        .tint(FVTheme.lime)
                        .frame(maxWidth: 320)
                } else if let seconds = record.metrics?.totalSeconds {
                    Text("Completed in \(seconds.formatted(.number.precision(.fractionLength(1)))) seconds")
                        .font(.system(size: 11.5))
                        .foregroundStyle(FVTheme.faint)
                }
            }
            Spacer()
            if record.status == .running {
                Button(role: .destructive) { model.cancelGeneration() } label: {
                    Label("Stop", systemImage: "stop.fill")
                }
                .buttonStyle(FVQuietButtonStyle(danger: true))
            } else if record.status == .completed, let url = record.outputURL {
                Button { model.reveal(record) } label: { Label("Show in Finder", systemImage: "folder") }
                    .buttonStyle(FVQuietButtonStyle())
                Button { model.export(record) } label: { Label("Export", systemImage: "arrow.down.to.line") }
                    .buttonStyle(FVQuietButtonStyle())
                ShareLink(item: url) { Label("Share", systemImage: "square.and.arrow.up") }
                    .buttonStyle(FVProminentButtonStyle())
            }
        }
        .frame(minHeight: 46)
    }

    private var footerTitle: String {
        switch record.status {
        case .running: record.previewPath == nil ? "Preparing your first preview" : "Refining the final video"
        case .completed: "Your video is ready"
        case .failed: "Generation stopped"
        case .cancelled: "Generation cancelled"
        case .queued: "Waiting to begin"
        }
    }
}

private struct StatusPill: View {
    let record: GenerationRecord

    var body: some View {
        HStack(spacing: 7) {
            Circle().fill(color).frame(width: 6, height: 6)
            Text(label)
                .font(.system(size: 10.5, weight: .medium))
        }
        .foregroundStyle(FVTheme.muted)
        .padding(.horizontal, 11)
        .frame(height: 28)
        .fvGlassCapsule(tint: color.opacity(0.30), fallback: Color.white.opacity(0.06))
    }

    private var color: Color {
        switch record.status {
        case .running, .queued: FVTheme.amber
        case .completed: FVTheme.lime
        case .failed, .cancelled: FVTheme.red
        }
    }

    private var label: String {
        switch record.status {
        case .running: record.phase
        case .completed: "Complete"
        case .failed: "Failed"
        case .cancelled: "Cancelled"
        case .queued: "Queued"
        }
    }
}

private struct LibraryView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            PageTitle(
                title: "Library",
                detail: model.records.isEmpty ? "Your finished videos will appear here." : "\(model.records.count) local generations"
            ) {
                Button { model.startNewGeneration() } label: { Label("New video", systemImage: "plus") }
                    .buttonStyle(FVProminentButtonStyle())
            }

            if model.records.isEmpty {
                ContentUnavailableView {
                    Label("No videos yet", systemImage: "film.stack")
                } description: {
                    Text("Create your first FastWan QAD video.")
                } actions: {
                    Button("Create video") { model.startNewGeneration() }
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                HStack(spacing: 0) {
                    ScrollView {
                        LazyVGrid(columns: [GridItem(.adaptive(minimum: 245), spacing: 16)], spacing: 20) {
                            ForEach(model.records) { record in
                                LibraryCard(record: record, selected: model.selectedRecordID == record.id) {
                                    model.selectedRecordID = record.id
                                }
                            }
                        }
                        .padding(.horizontal, 28)
                        .padding(.vertical, 8)
                        .padding(.bottom, 30)
                    }
                    Divider().overlay(FVTheme.line)
                    LibraryDetail(record: model.activeRecord)
                        .frame(minWidth: 330, idealWidth: 390, maxWidth: 430)
                }
            }
        }
    }
}

private struct LibraryCard: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var hovering = false
    let record: GenerationRecord
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 11) {
                VideoThumbnailView(url: record.outputURL)
                    .aspectRatio(832.0 / 480.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay {
                        if hovering {
                            Image(systemName: "play.fill")
                                .font(.system(size: 15, weight: .semibold))
                                .foregroundStyle(.white.opacity(0.94))
                                .frame(width: 46, height: 46)
                                .fvLiquidGlass(cornerRadius: 23)
                                .transition(.opacity.combined(with: .scale(scale: 0.82)))
                        }
                    }
                    .overlay(alignment: .topTrailing) {
                        Text(record.settings.mode == .fast
                            ? "\(record.settings.variant.displayName) · FAST"
                            : record.settings.variant.displayName)
                            .font(.system(size: 9, weight: .semibold))
                            .padding(.horizontal, 8)
                            .frame(height: 23)
                            .background {
                                if #available(macOS 26.0, *) {
                                    Color.clear
                                        .glassEffect(.regular, in: .capsule)
                                } else {
                                    Capsule()
                                        .fill(.ultraThinMaterial)
                                }
                            }
                            .padding(8)
                    }
                Text(record.prompt)
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(FVTheme.text)
                    .lineLimit(2)
                    .multilineTextAlignment(.leading)
                HStack {
                    Text(record.createdAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                    Spacer()
                    Text(record.status == .completed ? "\(record.settings.frames) frames" : record.status.rawValue.capitalized)
                }
                .font(.system(size: 10.5))
                .foregroundStyle(FVTheme.faint)
            }
            .padding(10)
            .background {
                if selected {
                    if #available(macOS 26.0, *) {
                        Color.clear
                            .glassEffect(.regular.tint(FVTheme.lime.opacity(0.10)), in: .rect(cornerRadius: 16))
                    } else {
                        RoundedRectangle(cornerRadius: 16, style: .continuous)
                            .fill(FVTheme.surfaceStrong)
                    }
                }
            }
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(selected ? FVTheme.lime.opacity(0.45) : Color.clear)
            )
        }
        .buttonStyle(.plain)
        .scaleEffect(hovering && !reduceMotion ? 1.015 : 1)
        .shadow(color: Color.black.opacity(hovering ? 0.32 : 0), radius: 18, y: 9)
        .onHover { hovering = $0 }
        .animation(.spring(response: 0.32, dampingFraction: 0.8), value: hovering)
    }
}

private struct VideoThumbnailView: View {
    let url: URL?
    @State private var image: NSImage?

    var body: some View {
        ZStack {
            Rectangle().fill(Color.white.opacity(0.045))
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
            } else {
                Image(systemName: "film")
                    .font(.system(size: 24, weight: .light))
                    .foregroundStyle(FVTheme.faint)
            }
        }
        .clipped()
        .task(id: url) { await loadThumbnail() }
    }

    private func loadThumbnail() async {
        guard let url, FileManager.default.fileExists(atPath: url.path) else { return }
        let result = await Task.detached(priority: .utility) { () -> NSImage? in
            let asset = AVURLAsset(url: url)
            let generator = AVAssetImageGenerator(asset: asset)
            generator.appliesPreferredTrackTransform = true
            generator.maximumSize = CGSize(width: 640, height: 360)
            guard let cgImage = try? generator.copyCGImage(at: CMTime(seconds: 0.35, preferredTimescale: 600), actualTime: nil)
            else { return nil }
            return NSImage(cgImage: cgImage, size: .zero)
        }.value
        image = result
    }
}

private struct LibraryDetail: View {
    @EnvironmentObject private var model: AppModel
    let record: GenerationRecord?

    var body: some View {
        Group {
            if let record {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        if let url = record.playbackURL, FileManager.default.fileExists(atPath: url.path) {
                            VideoSurface(url: url)
                                .aspectRatio(832.0 / 480.0, contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        } else {
                            VideoThumbnailView(url: record.outputURL)
                                .aspectRatio(832.0 / 480.0, contentMode: .fit)
                                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                        Text(record.prompt)
                            .font(.system(size: 17, weight: .medium, design: .rounded))
                            .lineLimit(7)
                            .textSelection(.enabled)
                        Divider().overlay(FVTheme.line)
                        DetailRow(label: "Created", value: record.createdAt.formatted(date: .abbreviated, time: .shortened))
                        DetailRow(label: "Model", value: record.settings.variant == .ema ? "EMA · Recommended" : "RAW")
                        DetailRow(label: "Generation", value: record.settings.mode == .fast ? "Fast · RIFE 2×" : "Full · Native frames")
                        DetailRow(label: "Format", value: "\(record.settings.resolutionLabel) · \(record.settings.frames) frames")
                        if let seconds = record.metrics?.totalSeconds {
                            DetailRow(label: "Render time", value: "\(seconds.formatted(.number.precision(.fractionLength(1)))) seconds")
                        }
                        if record.status == .completed, let url = record.outputURL {
                            HStack(spacing: 8) {
                                Button { model.showInCreate(record) } label: { Label("Open", systemImage: "arrow.up.left.and.arrow.down.right") }
                                    .buttonStyle(FVProminentButtonStyle())
                                Button { model.reveal(record) } label: { Label("Finder", systemImage: "folder") }
                                    .buttonStyle(FVQuietButtonStyle())
                                Button { model.export(record) } label: { Label("Export", systemImage: "arrow.down.to.line") }
                                    .buttonStyle(FVQuietButtonStyle())
                                ShareLink(item: url) { Label("Share", systemImage: "square.and.arrow.up") }
                                    .buttonStyle(FVQuietButtonStyle())
                            }
                        }
                        Button(role: .destructive) { model.delete(record) } label: {
                            Label("Delete generation", systemImage: "trash")
                        }
                        .buttonStyle(.plain)
                        .font(.system(size: 11.5))
                        .foregroundStyle(FVTheme.red)
                    }
                    .padding(24)
                }
            } else {
                Text("Select a video")
                    .foregroundStyle(FVTheme.faint)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Color.black.opacity(0.12))
    }
}

private struct DetailRow: View {
    let label: String
    let value: String
    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(FVTheme.faint)
            Spacer()
            Text(value).foregroundStyle(FVTheme.muted).multilineTextAlignment(.trailing)
        }
        .font(.system(size: 11.5))
    }
}

private struct SetupView: View {
    @EnvironmentObject private var model: AppModel
    @AppStorage("fastvideo.onboarding.completed") private var onboardingCompleted = true
    @State private var showDeveloperOptions = false
    @State private var pendingReset: ResetScope?
    @State private var showFullResetConfirmation = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 30) {
                PageTitle(
                    title: "Models & Runtime",
                    detail: "Everything FastWan QAD needs to create video on this Mac."
                ) {
                    Button { Task { await model.refreshRuntime() } } label: {
                        Label("Check again", systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(FVQuietButtonStyle())
                }

                if case let .error(message) = model.runtimeHealth.state {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(FVTheme.red)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Runtime check failed")
                                .font(.system(size: 13, weight: .medium))
                            Text(message)
                                .font(.system(size: 11.5))
                                .foregroundStyle(FVTheme.faint)
                                .textSelection(.enabled)
                        }
                    }
                    .padding(16)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 13).fill(FVTheme.red.opacity(0.08)))
                    .overlay(RoundedRectangle(cornerRadius: 13).stroke(FVTheme.red.opacity(0.24)))
                }

                SettingsSection(title: "Models", detail: "EMA is recommended for most videos. Add RAW only when you want to compare checkpoints.") {
                    ModelInstallRow(variant: .ema, recommended: true)
                    Divider().overlay(FVTheme.line)
                    ModelInstallRow(variant: .raw, recommended: false)
                }

                SettingsSection(title: "This Mac", detail: "The runtime stays local and uses Apple silicon directly.") {
                    RuntimeRow(
                        symbol: "apple.logo",
                        title: "Apple silicon",
                        detail: model.runtimeHealth.platformSupported ? "Supported" : "An Apple silicon Mac is required",
                        ready: model.runtimeHealth.platformSupported
                    )
                    Divider().overlay(FVTheme.line)
                    RuntimeRow(
                        symbol: "cpu",
                        title: "MLX and Metal",
                        detail: runtimeDetail,
                        ready: model.runtimeHealth.mlxAvailable && model.runtimeHealth.torchAvailable,
                        actionTitle: "Install",
                        action: model.installRuntime
                    )
                    Divider().overlay(FVTheme.line)
                    RuntimeRow(
                        symbol: "bolt.fill",
                        title: "Fast generation",
                        detail: model.runtimeHealth.rifeAvailable
                            ? "RIFE motion interpolation ready"
                            : "Install the latest runtime for ~2.7× faster generation",
                        ready: model.runtimeHealth.rifeAvailable,
                        actionTitle: "Install",
                        action: model.installFastMode
                    )
                    Divider().overlay(FVTheme.line)
                    RuntimeRow(
                        symbol: "film",
                        title: "Video export",
                        detail: model.runtimeHealth.ffmpegAvailable ? "Ready" : "Install the video exporter",
                        ready: model.runtimeHealth.ffmpegAvailable,
                        actionTitle: "Install",
                        action: model.installFFmpeg
                    )
                }

                SettingsSection(
                    title: "Uninstall & Reset",
                    detail: "Remove anything FastWan QAD has added to this Mac. Nothing outside these folders is ever touched."
                ) {
                    ResetRow(scope: .models, sizeText: model.resetSizes[.models] ?? "…") { pendingReset = .models }
                    Divider().overlay(FVTheme.line)
                    ResetRow(scope: .runtime, sizeText: model.resetSizes[.runtime] ?? "…") { pendingReset = .runtime }
                    Divider().overlay(FVTheme.line)
                    ResetRow(scope: .library, sizeText: model.resetSizes[.library] ?? "…") { pendingReset = .library }
                    Divider().overlay(FVTheme.line)
                    ResetRow(scope: .settings, sizeText: model.resetSizes[.settings] ?? "…") { pendingReset = .settings }
                    Divider().overlay(FVTheme.line)
                    HStack(spacing: 15) {
                        Image(systemName: "arrow.uturn.backward")
                            .font(.system(size: 16, weight: .medium))
                            .foregroundStyle(FVTheme.red)
                            .frame(width: 30)
                        VStack(alignment: .leading, spacing: 3) {
                            Text("Reset everything")
                                .font(.system(size: 14, weight: .medium, design: .rounded))
                            Text("Delete all of the above, then quit. Move FastWan QAD.app to the Trash to finish uninstalling.")
                                .font(.system(size: 11.5))
                                .foregroundStyle(FVTheme.faint)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                        Spacer()
                        Button("Reset & Quit…") { showFullResetConfirmation = true }
                            .buttonStyle(FVQuietButtonStyle(danger: true))
                    }
                    .padding(.vertical, 6)
                }

                DisclosureGroup("Developer options", isExpanded: $showDeveloperOptions) {
                    VStack(spacing: 14) {
                        PathRow(label: "FastVideo source", value: $model.configuration.repositoryRoot, choose: model.chooseRepositoryRoot)
                        PathRow(label: "Python", value: $model.configuration.pythonExecutable, choose: model.choosePython)
                        PathRow(label: "Shared model", value: $model.configuration.modelRoot, choose: model.chooseModelRoot)
                        HStack(spacing: 10) {
                            VStack(alignment: .leading, spacing: 6) {
                                Text("RAW checkpoint").font(.system(size: 10.5)).foregroundStyle(FVTheme.faint)
                                TextField("Auto-detect", text: $model.configuration.rawCheckpoint).textFieldStyle(FVTextFieldStyle())
                            }
                            VStack(alignment: .leading, spacing: 6) {
                                Text("EMA checkpoint").font(.system(size: 10.5)).foregroundStyle(FVTheme.faint)
                                TextField("Auto-detect", text: $model.configuration.emaCheckpoint).textFieldStyle(FVTextFieldStyle())
                            }
                        }
                        HStack {
                            Button("Replay onboarding") { onboardingCompleted = false }
                                .buttonStyle(FVQuietButtonStyle())
                            Spacer()
                            Button("Save changes") { model.saveConfiguration() }
                                .buttonStyle(FVProminentButtonStyle())
                        }
                    }
                    .padding(.top, 18)
                }
                .font(.system(size: 12.5, weight: .medium))
                .foregroundStyle(FVTheme.muted)
                .padding(20)
                .fvGlassSurface(cornerRadius: 16)

                if !model.setupLog.isEmpty {
                    DisclosureGroup("Installation details") {
                        Text(model.setupLog.suffix(100).joined(separator: "\n"))
                            .font(.system(size: 10, design: .monospaced))
                            .foregroundStyle(FVTheme.faint)
                            .textSelection(.enabled)
                            .padding(.top, 12)
                    }
                    .font(.system(size: 11.5))
                    .foregroundStyle(FVTheme.faint)
                }
            }
            .frame(maxWidth: 900)
            .padding(.horizontal, 38)
            .padding(.bottom, 50)
        }
        .onAppear { model.refreshResetSizes() }
        .alert(item: $pendingReset) { scope in
            Alert(
                title: Text(scope.confirmTitle),
                message: Text(scope.confirmDetail),
                primaryButton: .destructive(Text(scope.actionLabel)) { model.performReset(scope) },
                secondaryButton: .cancel()
            )
        }
        .alert("Reset FastWan QAD?", isPresented: $showFullResetConfirmation) {
            Button("Delete Everything & Quit", role: .destructive) { model.resetEverythingAndQuit() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Models, the runtime environment, your entire library, and all settings will be permanently deleted from this Mac.")
        }
    }

    private var runtimeDetail: String {
        guard model.runtimeHealth.mlxAvailable && model.runtimeHealth.torchAvailable else {
            return "Install the local runtime"
        }
        return model.runtimeHealth.mpsAvailable
            ? "MLX Metal ready · MPS auxiliaries"
            : "MLX Metal ready · CPU auxiliaries"
    }
}

private struct ModelInstallRow: View {
    @EnvironmentObject private var model: AppModel
    let variant: ModelVariant
    let recommended: Bool

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: variant == .ema ? "waveform.path.ecg" : "point.3.connected.trianglepath.dotted")
                .font(.system(size: 17, weight: .medium))
                .foregroundStyle(installed ? FVTheme.lime : FVTheme.muted)
                .frame(width: 30)
            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 8) {
                    Text("\(variant.displayName) model")
                        .font(.system(size: 14, weight: .medium, design: .rounded))
                    if recommended {
                        Text("RECOMMENDED")
                            .font(.system(size: 8.5, weight: .bold))
                            .tracking(0.7)
                            .foregroundStyle(Color.black.opacity(0.78))
                            .padding(.horizontal, 7)
                            .frame(height: 19)
                            .background(Capsule().fill(FVTheme.accentGradient))
                    }
                }
                Text(variant.detail)
                    .font(.system(size: 11.5))
                    .foregroundStyle(FVTheme.faint)
            }
            Spacer()
            if model.installingVariant == variant {
                VStack(alignment: .trailing, spacing: 5) {
                    ProgressView(value: model.modelInstallProgress)
                        .frame(width: 120)
                        .tint(FVTheme.lime)
                    Text("Downloading…")
                        .font(.system(size: 9.5))
                        .foregroundStyle(FVTheme.faint)
                }
            } else if installed {
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .font(.system(size: 11.5, weight: .medium))
                    .foregroundStyle(FVTheme.lime)
            } else {
                if recommended {
                    Button("Download EMA") { model.installModelWithRuntime(variant) }
                        .buttonStyle(FVProminentButtonStyle())
                        .disabled(model.isInstallingModel || model.isInstallingRuntime)
                } else {
                    Button("Download RAW") { model.installModelWithRuntime(variant) }
                        .buttonStyle(FVQuietButtonStyle())
                        .disabled(model.isInstallingModel || model.isInstallingRuntime)
                }
            }
        }
        .padding(.vertical, 6)
    }

    private var installed: Bool { model.runtimeHealth.variantAvailable(variant) }
}

private struct ResetRow: View {
    let scope: ResetScope
    let sizeText: String
    let action: () -> Void

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: scope.symbol)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(FVTheme.muted)
                .frame(width: 30)
            VStack(alignment: .leading, spacing: 3) {
                Text(scope.title)
                    .font(.system(size: 14, weight: .medium, design: .rounded))
                Text(scope.detail)
                    .font(.system(size: 11.5))
                    .foregroundStyle(FVTheme.faint)
            }
            Spacer()
            Text(sizeText)
                .font(.system(size: 11, design: .monospaced))
                .foregroundStyle(FVTheme.faint)
            Button(scope.actionLabel, action: action)
                .buttonStyle(FVQuietButtonStyle(danger: true))
        }
        .padding(.vertical, 6)
    }
}

private struct RuntimeRow: View {
    let symbol: String
    let title: String
    let detail: String
    let ready: Bool
    var actionTitle: String?
    var action: (() -> Void)?

    var body: some View {
        HStack(spacing: 15) {
            Image(systemName: symbol)
                .font(.system(size: 16, weight: .medium))
                .foregroundStyle(ready ? FVTheme.lime : FVTheme.muted)
                .frame(width: 30)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.system(size: 14, weight: .medium, design: .rounded))
                Text(detail).font(.system(size: 11.5)).foregroundStyle(FVTheme.faint)
            }
            Spacer()
            if ready {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(FVTheme.lime)
            } else if let actionTitle, let action {
                Button(actionTitle, action: action).buttonStyle(FVQuietButtonStyle())
            }
        }
        .padding(.vertical, 6)
    }
}

private struct SettingsSection<Content: View>: View {
    let title: String
    let detail: String
    @ViewBuilder let content: Content

    init(title: String, detail: String, @ViewBuilder content: () -> Content) {
        self.title = title
        self.detail = detail
        self.content = content()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 13) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.system(size: 18, weight: .semibold, design: .rounded))
                Text(detail).font(.system(size: 12)).foregroundStyle(FVTheme.faint)
            }
            VStack(spacing: 12) { content }
                .padding(20)
                .fvGlassSurface(cornerRadius: 16)
        }
    }
}

private struct PageTitle<Actions: View>: View {
    let title: String
    let detail: String
    @ViewBuilder let actions: Actions

    init(title: String, detail: String, @ViewBuilder actions: () -> Actions) {
        self.title = title
        self.detail = detail
        self.actions = actions()
    }

    var body: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.system(size: 31, weight: .semibold, design: .rounded))
                    .tracking(-0.8)
                Text(detail)
                    .font(.system(size: 12.5))
                    .foregroundStyle(FVTheme.faint)
            }
            Spacer()
            actions
        }
        .padding(.horizontal, 30)
        .padding(.top, 45)
        .padding(.bottom, 24)
    }
}

private struct PathRow: View {
    let label: String
    @Binding var value: String
    let choose: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(label).font(.system(size: 10.5)).foregroundStyle(FVTheme.faint)
            HStack(spacing: 8) {
                TextField(label, text: $value).textFieldStyle(FVTextFieldStyle())
                Button("Choose…", action: choose).buttonStyle(FVQuietButtonStyle())
            }
        }
    }
}

/// A tiny diffusion story, told softly: a painted miniature scene that
/// resolves from a deep blur into clarity, settles, then fades so the loop can
/// begin again. No flicker, no cells — just blur becoming light.
struct DenoisePreview: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.periodic(from: .now, by: reduceMotion ? 3600 : 1.0 / 30.0)) { timeline in
            let phase = Self.phase(at: timeline.date.timeIntervalSinceReferenceDate, reduceMotion: reduceMotion)
            scene
                .blur(radius: phase.blur)
                .scaleEffect(phase.scale)
                .overlay(Color.black.opacity(phase.dim))
                .opacity(phase.opacity)
        }
        .accessibilityHidden(true)
    }

    /// One loop: unblur over the first 55%, hold sharp, fade out at the end so
    /// the restart is never visible.
    private struct Phase {
        var blur: CGFloat
        var scale: CGFloat
        var dim: Double
        var opacity: Double
    }

    private static func phase(at tick: TimeInterval, reduceMotion: Bool) -> Phase {
        guard !reduceMotion else { return Phase(blur: 0, scale: 1, dim: 0, opacity: 1) }
        let t = tick.truncatingRemainder(dividingBy: 8.0) / 8.0
        let resolve = easeInOut(min(1, t / 0.55))
        let fadeOut = t > 0.88 ? easeInOut((t - 0.88) / 0.12) : 0
        return Phase(
            blur: (1 - resolve) * 16,
            scale: 1.05 - resolve * 0.05,
            dim: (1 - resolve) * 0.30,
            opacity: 1 - fadeOut
        )
    }

    private static func easeInOut(_ x: Double) -> Double {
        x < 0.5 ? 2 * x * x : 1 - pow(-2 * x + 2, 2) / 2
    }

    /// A miniature landscape: teal sky melting into a warm horizon, a low sun,
    /// and a dark ridge in front.
    private var scene: some View {
        Canvas { context, size in
            let horizon = size.height * 0.62
            context.fill(
                Path(CGRect(x: 0, y: 0, width: size.width, height: horizon)),
                with: .linearGradient(
                    Gradient(colors: [
                        Color(red: 0.07, green: 0.23, blue: 0.25),
                        Color(red: 0.16, green: 0.36, blue: 0.33),
                        Color(red: 0.83, green: 0.52, blue: 0.28),
                    ]),
                    startPoint: .zero,
                    endPoint: CGPoint(x: 0, y: horizon)
                )
            )
            let sunCenter = CGPoint(x: size.width * 0.68, y: horizon * 0.72)
            let sunRadius = size.width * 0.075
            context.fill(
                Path(ellipseIn: CGRect(
                    x: sunCenter.x - sunRadius,
                    y: sunCenter.y - sunRadius,
                    width: sunRadius * 2,
                    height: sunRadius * 2
                )),
                with: .radialGradient(
                    Gradient(colors: [
                        Color(red: 1.0, green: 0.90, blue: 0.62),
                        Color(red: 1.0, green: 0.78, blue: 0.45).opacity(0.55),
                        .clear,
                    ]),
                    center: sunCenter,
                    startRadius: 0,
                    endRadius: sunRadius
                )
            )
            context.fill(
                Path(CGRect(x: 0, y: horizon, width: size.width, height: size.height - horizon)),
                with: .color(Color(red: 0.05, green: 0.09, blue: 0.08))
            )
            var ridge = Path()
            ridge.move(to: CGPoint(x: 0, y: size.height))
            for x in stride(from: 0, through: size.width, by: 4) {
                let u = x / size.width
                ridge.addLine(to: CGPoint(x: x, y: size.height * (0.70 + 0.055 * sin(u * 9 + 1.4))))
            }
            ridge.addLine(to: CGPoint(x: size.width, y: size.height))
            ridge.closeSubpath()
            context.fill(ridge, with: .color(Color(red: 0.13, green: 0.20, blue: 0.17)))
        }
    }
}

struct ASCIIFieldView: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    var body: some View {
        TimelineView(.periodic(from: .now, by: reduceMotion ? 2 : 0.16)) { timeline in
            GeometryReader { proxy in
                Text(field(for: timeline.date.timeIntervalSinceReferenceDate, size: proxy.size))
                    .font(.system(size: 8, weight: .regular, design: .monospaced))
                    .lineSpacing(0.8)
                    .foregroundStyle(FVTheme.lime.opacity(0.48))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .clipped()
            }
        }
        .accessibilityHidden(true)
    }

    private func field(for time: TimeInterval, size: CGSize) -> String {
        let width = max(48, min(150, Int(size.width / 7.2)))
        let height = max(20, min(70, Int(size.height / 9.0)))
        let glyphs = Array("  ..::--==++**##%@")
        let t = reduceMotion ? 0.3 : time * 0.31
        return (0..<height).map { row in
            String((0..<width).map { column in
                let x = Double(column) / Double(width) * 6.4 - 3.2
                let y = Double(row) / Double(height) * 3.0 - 1.5
                let wave = sin(x * 1.7 + t) + cos(y * 3.1 - t * 0.72)
                let orbit = sin(hypot(x - sin(t), y - cos(t * 0.8)) * 4.4 - t * 2)
                let vignette = max(0, 1 - (x * x / 13 + y * y / 3.2))
                let value = max(0, min(0.999, (wave + orbit + 3) / 6 * vignette))
                return glyphs[Int(value * Double(glyphs.count))]
            })
        }.joined(separator: "\n")
    }
}

private struct VideoSurface: NSViewRepresentable {
    let url: URL

    final class Coordinator { var url: URL? }
    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> AVPlayerView {
        let view = AVPlayerView()
        view.controlsStyle = .floating
        updatePlayer(in: view, context: context)
        return view
    }

    func updateNSView(_ view: AVPlayerView, context: Context) { updatePlayer(in: view, context: context) }

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

private struct GeneratingPlaceholder: View {
    let record: GenerationRecord
    var body: some View {
        ZStack {
            DenoisePreview()
            LinearGradient(
                colors: [Color.black.opacity(0.10), Color.black.opacity(0.58)],
                startPoint: .top,
                endPoint: .bottom
            )
            VStack(spacing: 12) {
                Spacer()
                Text(record.phase)
                    .font(.system(size: 15, weight: .semibold, design: .rounded))
                    .foregroundStyle(.white.opacity(0.94))
                ProgressView(value: record.progress)
                    .tint(FVTheme.lime)
                    .frame(maxWidth: 220)
                Text("First light appears here while the model keeps refining.")
                    .font(.system(size: 11))
                    .foregroundStyle(.white.opacity(0.62))
            }
            .padding(.bottom, 24)
        }
    }
}

private struct FailureView: View {
    let record: GenerationRecord
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle").font(.system(size: 24)).foregroundStyle(FVTheme.red)
            Text("Generation could not finish").font(.system(size: 15, weight: .medium))
            Text(record.error ?? "Review Models & Runtime, then try again.")
                .font(.system(size: 11.5)).foregroundStyle(FVTheme.faint).multilineTextAlignment(.center)
        }
        .padding(30)
    }
}

private struct LiveBadge: View {
    var body: some View {
        Label("Live preview", systemImage: "sparkles")
            .font(.system(size: 10, weight: .semibold))
            .foregroundStyle(Color.black.opacity(0.82))
            .padding(.horizontal, 10)
            .frame(height: 28)
            .fvGlassCapsule(tint: FVTheme.lime, fallback: FVTheme.lime)
    }
}

private struct FVProminentButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        if #available(macOS 26.0, *) {
            configuration.label
                .font(.system(size: 11.5, weight: .semibold))
                .foregroundStyle(Color.black.opacity(0.84))
                .padding(.horizontal, 14)
                .frame(minHeight: 34)
                .glassEffect(.regular.tint(FVTheme.lime).interactive(true), in: .capsule)
                .opacity(configuration.isPressed ? 0.85 : 1)
        } else {
            configuration.label
                .font(.system(size: 11.5, weight: .semibold))
                .foregroundStyle(Color.black.opacity(0.84))
                .padding(.horizontal, 14)
                .frame(minHeight: 34)
                .background(Capsule().fill(FVTheme.lime.opacity(configuration.isPressed ? 0.72 : 1)))
        }
    }
}

private struct FVQuietButtonStyle: ButtonStyle {
    var danger = false
    func makeBody(configuration: Configuration) -> some View {
        if #available(macOS 26.0, *) {
            configuration.label
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(danger ? FVTheme.red : FVTheme.muted)
                .padding(.horizontal, 13)
                .frame(minHeight: 34)
                .glassEffect(.regular.interactive(true), in: .capsule)
                .opacity(configuration.isPressed ? 0.78 : 1)
        } else {
            configuration.label
                .font(.system(size: 11.5, weight: .medium))
                .foregroundStyle(danger ? FVTheme.red : FVTheme.muted)
                .padding(.horizontal, 13)
                .frame(minHeight: 34)
                .background(Capsule().fill(configuration.isPressed ? FVTheme.surfaceStrong : FVTheme.surface))
                .overlay(Capsule().stroke(FVTheme.line))
        }
    }
}

private struct FVTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .font(.system(size: 10.5, design: .monospaced))
            .padding(.horizontal, 11)
            .frame(height: 34)
            .background(RoundedRectangle(cornerRadius: 7).fill(Color.black.opacity(0.18)))
            .overlay(RoundedRectangle(cornerRadius: 7).stroke(FVTheme.line))
    }
}
