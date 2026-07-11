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
    static let amber = Color(red: 0.96, green: 0.70, blue: 0.32)
    static let red = Color(red: 0.96, green: 0.36, blue: 0.32)
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
                    .fill(FVTheme.lime)
                    .frame(width: 32, height: 32)
                    .overlay {
                        Text("FQ")
                            .font(.system(size: 10, weight: .black, design: .rounded))
                            .foregroundStyle(Color.black.opacity(0.82))
                    }
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
                        .background(
                            RoundedRectangle(cornerRadius: 9, style: .continuous)
                                .fill(model.section == section ? Color.white.opacity(0.085) : .clear)
                        )
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
            .padding(.horizontal, 17)
            .padding(.vertical, 16)
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
            FVTheme.background.ignoresSafeArea()
            ASCIIFieldView()
                .opacity(model.createRecord == nil ? 0.34 : 0.12)
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
                    .foregroundStyle(FVTheme.text)
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
                    Image(systemName: "arrow.up")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(Color.black.opacity(0.82))
                        .frame(width: 38, height: 38)
                        .background(Circle().fill(FVTheme.lime))
                }
                .buttonStyle(.plain)
                .disabled(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .opacity(model.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? 0.42 : 1)
                .help("Generate video")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .frame(maxWidth: 760, minHeight: 64)
            .fvLiquidGlass(cornerRadius: 22, tint: Color.white.opacity(0.015), interactive: true)
            .shadow(color: Color.black.opacity(0.25), radius: 30, y: 16)

            Spacer(minLength: 120)
        }
        .padding(.horizontal, 48)
        .onAppear { promptFocused = true }
    }
}

private struct GenerationOptionsMenu: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        Menu {
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
            Image(systemName: "slider.horizontal.3")
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(FVTheme.muted)
                .frame(width: 36, height: 36)
                .background(Circle().fill(Color.white.opacity(0.065)))
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .help("Generation options")
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
                .overlay(RoundedRectangle(cornerRadius: 18).stroke(FVTheme.line))
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
        .background(Capsule().fill(Color.white.opacity(0.06)))
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
    let record: GenerationRecord
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            VStack(alignment: .leading, spacing: 11) {
                VideoThumbnailView(url: record.outputURL)
                    .aspectRatio(832.0 / 480.0, contentMode: .fit)
                    .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    .overlay(alignment: .topTrailing) {
                        Text(record.settings.variant.displayName)
                            .font(.system(size: 9, weight: .semibold))
                            .padding(.horizontal, 8)
                            .frame(height: 23)
                            .background(.ultraThinMaterial, in: Capsule())
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
            .background(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .fill(selected ? FVTheme.surfaceStrong : Color.clear)
            )
            .overlay(
                RoundedRectangle(cornerRadius: 16, style: .continuous)
                    .stroke(selected ? FVTheme.lime.opacity(0.45) : Color.clear)
            )
        }
        .buttonStyle(.plain)
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
                        detail: model.runtimeHealth.mlxAvailable && model.runtimeHealth.mpsAvailable
                            ? "Ready for local generation"
                            : "Install the local runtime",
                        ready: model.runtimeHealth.mlxAvailable && model.runtimeHealth.mpsAvailable,
                        actionTitle: "Install",
                        action: model.installRuntime
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
                .background(RoundedRectangle(cornerRadius: 16).fill(FVTheme.surface))

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
                            .background(Capsule().fill(FVTheme.lime))
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
                .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(FVTheme.surface))
                .overlay(RoundedRectangle(cornerRadius: 16).stroke(FVTheme.line))
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
        VStack(spacing: 18) {
            ProgressView(value: record.progress)
                .progressViewStyle(.circular)
                .tint(FVTheme.lime)
                .scaleEffect(1.2)
            Text(record.phase)
                .font(.system(size: 14, weight: .medium, design: .rounded))
            Text("The first preview will appear here while the model keeps refining.")
                .font(.system(size: 11.5))
                .foregroundStyle(FVTheme.faint)
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
            .background(Capsule().fill(FVTheme.lime))
    }
}

private struct FVProminentButtonStyle: ButtonStyle {
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 11.5, weight: .semibold))
            .foregroundStyle(Color.black.opacity(0.84))
            .padding(.horizontal, 14)
            .frame(minHeight: 34)
            .background(Capsule().fill(FVTheme.lime.opacity(configuration.isPressed ? 0.72 : 1)))
    }
}

private struct FVQuietButtonStyle: ButtonStyle {
    var danger = false
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 11.5, weight: .medium))
            .foregroundStyle(danger ? FVTheme.red : FVTheme.muted)
            .padding(.horizontal, 13)
            .frame(minHeight: 34)
            .background(Capsule().fill(configuration.isPressed ? FVTheme.surfaceStrong : FVTheme.surface))
            .overlay(Capsule().stroke(FVTheme.line))
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
