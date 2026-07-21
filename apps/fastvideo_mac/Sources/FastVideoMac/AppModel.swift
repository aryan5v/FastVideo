import AppKit
import Foundation
import UserNotifications

@MainActor
final class AppModel: ObservableObject {
    @Published var section: AppSection = .create
    @Published var records: [GenerationRecord] = []
    @Published var selectedRecordID: UUID?
    @Published var createRecordID: UUID?
    @Published var prompt = ""
    @Published var generationSettings = GenerationSettings()
    @Published var runtimeHealth = RuntimeHealth()
    @Published var configuration: RuntimeConfiguration
    @Published var setupLog: [String] = []
    @Published var isInstallingRuntime = false
    @Published var isInstallingModel = false
    @Published var installingVariant: ModelVariant?
    @Published var modelInstallProgress: Double?
    @Published var alertMessage: String?

    private let library: GenerationLibrary
    private let generationProcess = ProcessDriver()
    private let utilityProcess = ProcessDriver()
    private var generationActivity: NSObjectProtocol?
    private var generationCompletedEvent = false

    var activeRecord: GenerationRecord? {
        if let selectedRecordID, let selected = records.first(where: { $0.id == selectedRecordID }) {
            return selected
        }
        return records.first
    }

    var createRecord: GenerationRecord? {
        guard let createRecordID else { return nil }
        return records.first { $0.id == createRecordID }
    }

    var isGenerating: Bool { records.contains { $0.status == .running } }

    init(library: GenerationLibrary = GenerationLibrary(), defaults: UserDefaults = .standard) {
        self.library = library
        if let data = defaults.data(forKey: "fastvideo.runtime.configuration"),
           let stored = try? JSONDecoder().decode(RuntimeConfiguration.self, from: data) {
            configuration = stored
        } else {
            configuration = .defaults()
        }
        configuration.adoptBundledRuntime()
        configuration.adoptDetectedLocalArtifacts()
        if let data = try? JSONEncoder().encode(configuration) {
            defaults.set(data, forKey: "fastvideo.runtime.configuration")
        }
        do {
            records = try library.load()
            selectedRecordID = records.first?.id
        } catch {
            records = []
            alertMessage = "Could not load generation history: \(error.localizedDescription)"
        }
        Task { await refreshRuntime() }
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { _, _ in }
    }

    func saveConfiguration() {
        if let data = try? JSONEncoder().encode(configuration) {
            UserDefaults.standard.set(data, forKey: "fastvideo.runtime.configuration")
        }
        Task { await refreshRuntime() }
    }

    func refreshRuntime() async {
        runtimeHealth.state = .checking
        guard FileManager.default.isExecutableFile(atPath: configuration.pythonExecutable) else {
            runtimeHealth.state = .needsSetup
            return
        }
        guard FileManager.default.fileExists(atPath: configuration.bridgePath) else {
            runtimeHealth.state = .error("The local runtime is incomplete. Reinstall FastWan QAD or review Developer options.")
            return
        }
        var arguments = [
            configuration.bridgePath,
            "diagnose",
            "--model-root", configuration.modelRoot,
        ]
        if !configuration.rawCheckpoint.isEmpty {
            arguments += ["--raw-checkpoint", configuration.rawCheckpoint]
        }
        if !configuration.emaCheckpoint.isEmpty {
            arguments += ["--ema-checkpoint", configuration.emaCheckpoint]
        }
        do {
            let result = try await ProcessDriver.runAndCollect(
                executable: configuration.pythonExecutable,
                arguments: arguments,
                currentDirectory: configuration.repositoryRoot
            )
            guard result.status == 0,
                  let line = result.output.split(separator: "\n").last,
                  let data = String(line).data(using: .utf8),
                  let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                runtimeHealth.state = .error(result.output.isEmpty ? "Runtime diagnosis failed." : result.output)
                return
            }
            runtimeHealth = Self.runtimeHealth(from: object)
            if !runtimeHealth.variantAvailable(generationSettings.variant),
               let first = ModelVariant.allCases.first(where: runtimeHealth.variantAvailable) {
                generationSettings.variant = first
            }
            if generationSettings.mode == .fast, !runtimeHealth.rifeAvailable {
                generationSettings.mode = .full
            }
        } catch {
            runtimeHealth.state = .error(error.localizedDescription)
        }
    }

    func installRuntime() {
        guard !isInstallingRuntime else { return }
        guard !configuration.uvExecutable.isEmpty else {
            alertMessage = "The runtime installer is unavailable. Install uv or select it in Developer options."
            return
        }
        isInstallingRuntime = true
        setupLog = ["Creating a managed Python 3.12 environment…"]
        Task {
            do {
                try await installRuntimeComponents()
                saveConfiguration()
            } catch {
                alertMessage = "Runtime installation failed: \(error.localizedDescription)"
            }
            isInstallingRuntime = false
            await refreshRuntime()
        }
    }

    func installFastMode() {
        guard !isInstallingRuntime, !isInstallingModel else { return }
        guard !configuration.uvExecutable.isEmpty else {
            alertMessage = "The bundled runtime installer is missing. Reinstall FastWan QAD."
            return
        }
        isInstallingRuntime = true
        setupLog = ["Preparing Fast generation…"]
        Task {
            do {
                try await installRuntimeComponents()
                let result = try await ProcessDriver.runAndCollect(
                    executable: configuration.pythonExecutable,
                    arguments: [
                        configuration.bridgePath,
                        "install-fast-mode",
                        "--catalog", configuration.modelCatalogPath,
                        "--model-root", configuration.modelRoot,
                    ],
                    currentDirectory: configuration.repositoryRoot,
                    onLine: { [weak self] line in Task { @MainActor in self?.setupLog.append(line) } }
                )
                guard result.status == 0 else { throw AppError.commandFailed(result.output) }
                saveConfiguration()
            } catch {
                alertMessage = "Fast generation setup failed: \(error.localizedDescription)"
            }
            isInstallingRuntime = false
            await refreshRuntime()
        }
    }

    func installModelWithRuntime(_ variant: ModelVariant) {
        guard !isInstallingRuntime, !isInstallingModel else { return }
        if FileManager.default.isExecutableFile(atPath: configuration.pythonExecutable) {
            installModel(variant)
            return
        }
        guard !configuration.uvExecutable.isEmpty else {
            alertMessage = "The bundled runtime installer is missing. Reinstall FastWan QAD."
            return
        }
        isInstallingRuntime = true
        setupLog = ["Preparing the local Apple silicon runtime…"]
        Task {
            do {
                try await installRuntimeComponents()
                saveConfiguration()
                isInstallingRuntime = false
                await refreshRuntime()
                installModel(variant)
            } catch {
                isInstallingRuntime = false
                alertMessage = "Setup could not finish: \(error.localizedDescription)"
            }
        }
    }

    private func installRuntimeComponents() async throws {
        let repositoryRoot = configuration.repositoryRoot
        let venvRoot = RuntimeConfiguration.managedEnvironmentRoot()
        let venvPython = venvRoot.appendingPathComponent("bin/python").path
        let venv = try await ProcessDriver.runAndCollect(
            executable: configuration.uvExecutable,
            arguments: ["venv", "--python", "3.12", "--seed", venvRoot.path],
            currentDirectory: repositoryRoot,
            onLine: { [weak self] line in Task { @MainActor in self?.setupLog.append(line) } }
        )
        guard venv.status == 0 else { throw AppError.commandFailed(venv.output) }
        setupLog.append("Installing the FastWan QAD MLX runtime…")
        var installArguments = ["pip", "install", "--python", venvPython]
        let bundledSource = Bundle.main.resourceURL?
            .appendingPathComponent("fastvideo-source", isDirectory: true).standardizedFileURL.path
        if bundledSource != URL(fileURLWithPath: repositoryRoot).standardizedFileURL.path {
            installArguments.append("-e")
        }
        installArguments.append("\(repositoryRoot)[mlx]")
        let install = try await ProcessDriver.runAndCollect(
            executable: configuration.uvExecutable,
            arguments: installArguments,
            currentDirectory: repositoryRoot,
            onLine: { [weak self] line in Task { @MainActor in self?.setupLog.append(line) } }
        )
        guard install.status == 0 else { throw AppError.commandFailed(install.output) }
        configuration.pythonExecutable = venvPython
        generationSettings.mode = .fast
    }

    func installFFmpeg() {
        let brew = ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"].first {
            FileManager.default.isExecutableFile(atPath: $0)
        }
        guard let brew else {
            alertMessage = "The video exporter could not be installed automatically. Install ffmpeg or review Developer options."
            return
        }
        isInstallingRuntime = true
        setupLog.append("Installing ffmpeg with Homebrew…")
        Task {
            do {
                let result = try await ProcessDriver.runAndCollect(
                    executable: brew,
                    arguments: ["install", "ffmpeg"],
                    onLine: { [weak self] line in Task { @MainActor in self?.setupLog.append(line) } }
                )
                guard result.status == 0 else { throw AppError.commandFailed(result.output) }
            } catch {
                alertMessage = "ffmpeg installation failed: \(error.localizedDescription)"
            }
            isInstallingRuntime = false
            await refreshRuntime()
        }
    }

    func installModel(_ variant: ModelVariant = .ema) {
        guard !isInstallingModel else { return }
        guard FileManager.default.fileExists(atPath: configuration.modelCatalogPath) else {
            alertMessage = "The bundled FastWan QAD release catalog is missing. Reinstall the application."
            return
        }
        guard FileManager.default.isExecutableFile(atPath: configuration.pythonExecutable) else {
            alertMessage = "Prepare the local runtime before downloading a model."
            return
        }
        let installerPython = configuration.pythonExecutable
        let checkpointRoot = configuration.prepareInstallDestination(for: variant)
        saveConfiguration()
        isInstallingModel = true
        installingVariant = variant
        modelInstallProgress = nil
        setupLog.append("Preparing the \(variant.displayName) model…")
        let arguments = [
            configuration.bridgePath,
            "install-release",
            "--catalog", configuration.modelCatalogPath,
            "--variant", variant.rawValue,
            "--model-root", configuration.modelRoot,
            "--checkpoint-root", checkpointRoot,
        ]
        do {
            try utilityProcess.start(
                executable: installerPython,
                arguments: arguments,
                currentDirectory: configuration.repositoryRoot,
                onLine: { [weak self] line in
                    Task { @MainActor in self?.handleInstallLine(line) }
                },
                onTermination: { [weak self] status in
                    Task { @MainActor in
                        guard let self else { return }
                        self.isInstallingModel = false
                        self.installingVariant = nil
                        if status != 0 { self.alertMessage = "Model download stopped with code \(status)." }
                        await self.refreshRuntime()
                    }
                }
            )
        } catch {
            isInstallingModel = false
            installingVariant = nil
            alertMessage = "Could not start model download: \(error.localizedDescription)"
        }
    }

    func generate() {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { alertMessage = "Write a prompt first."; return }
        guard runtimeHealth.canGenerate else { section = .setup; return }
        guard !isGenerating else { return }
        guard generationSettings.mode != .fast || runtimeHealth.rifeAvailable else {
            alertMessage = "Fast mode needs the latest local runtime. Install it in Models & Runtime, or choose Full generation."
            section = .setup
            return
        }
        guard let checkpoint = runtimeHealth.checkpoint(for: generationSettings.variant) else {
            alertMessage = "The \(generationSettings.variant.label) checkpoint is not installed."
            return
        }

        let id = UUID()
        do {
            let outputURL = try library.outputURL(for: id)
            let requestURL = try library.requestURL(for: id)
            var request: [String: Any] = [
                "repo_root": configuration.repositoryRoot,
                "prompt": trimmed,
                "variant": generationSettings.variant.rawValue,
                "fast": generationSettings.mode == .fast,
                "model_root": configuration.modelRoot,
                "checkpoint_path": checkpoint,
                "output_path": outputURL.path,
                "height": generationSettings.height,
                "width": generationSettings.width,
                "num_frames": generationSettings.frames,
                "fps": generationSettings.fps,
                "seed": generationSettings.seed,
                "dmd_denoising_steps": generationSettings.dmdSteps,
                "taehv_parallel": generationSettings.parallelDecode,
            ]
            if let memoryLimitGiB = generationSettings.memoryLimitGiB {
                request["mlx_memory_limit_gib"] = memoryLimitGiB
            }
            let data = try JSONSerialization.data(withJSONObject: request, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: requestURL, options: Data.WritingOptions.atomic)

            let record = GenerationRecord(
                id: id,
                prompt: trimmed,
                createdAt: Date(),
                status: .running,
                settings: generationSettings,
                outputPath: outputURL.path,
                previewPath: nil,
                phase: "Preparing",
                progress: 0.02
            )
            records.insert(record, at: 0)
            selectedRecordID = id
            createRecordID = id
            try persist()
            generationCompletedEvent = false
            generationActivity = ProcessInfo.processInfo.beginActivity(
                options: [.userInitiated, .idleSystemSleepDisabled],
                reason: "FastWan QAD is generating a local video"
            )
            try generationProcess.start(
                executable: configuration.pythonExecutable,
                arguments: [configuration.bridgePath, "generate", "--request", requestURL.path],
                currentDirectory: configuration.repositoryRoot,
                onLine: { [weak self] line in Task { @MainActor in self?.handleGenerationLine(line, id: id) } },
                onTermination: { [weak self] status in Task { @MainActor in self?.finishGenerationProcess(id: id, status: status) } }
            )
        } catch {
            if let index = records.firstIndex(where: { $0.id == id }) {
                records[index].status = .failed
                records[index].error = error.localizedDescription
            }
            endGenerationActivity()
            alertMessage = "Could not start generation: \(error.localizedDescription)"
        }
    }

    func startNewGeneration() {
        createRecordID = nil
        prompt = ""
        section = .create
    }

    func showInCreate(_ record: GenerationRecord) {
        createRecordID = record.id
        section = .create
    }

    func cancelGeneration() {
        guard let index = records.firstIndex(where: { $0.status == .running }) else { return }
        records[index].status = .cancelled
        records[index].phase = "Cancelled"
        records[index].finishedAt = Date()
        generationProcess.cancel()
        try? persist()
        endGenerationActivity()
    }

    func delete(_ record: GenerationRecord) {
        do {
            try library.delete(record)
            records.removeAll { $0.id == record.id }
            selectedRecordID = records.first?.id
            try persist()
        } catch {
            alertMessage = "Could not delete generation: \(error.localizedDescription)"
        }
    }

    // MARK: - Uninstall & Reset

    @Published var resetSizes: [ResetScope: String] = [:]

    func refreshResetSizes() {
        let libraryRoot = library.baseURL
        Task.detached(priority: .utility) { [weak self] in
            var map: [ResetScope: String] = [:]
            for scope in ResetScope.allCases {
                if scope == .settings {
                    map[scope] = "Preferences"
                    continue
                }
                let bytes = ResetPlan.paths(for: scope, libraryRoot: libraryRoot)
                    .reduce(Int64(0)) { $0 + ResetPlan.byteCount(at: $1) }
                map[scope] = ResetPlan.formattedSize(bytes)
            }
            await self?.applyResetSizes(map)
        }
    }

    private func applyResetSizes(_ map: [ResetScope: String]) {
        resetSizes = map
    }

    /// Removes one category of app-owned data. Anything in flight is stopped
    /// first so no process holds files inside the folders being deleted.
    func performReset(_ scope: ResetScope) {
        stopBackgroundWork()
        if scope == .settings {
            for key in ResetPlan.defaultsKeys { UserDefaults.standard.removeObject(forKey: key) }
            configuration = .defaults()
            configuration.adoptBundledRuntime()
            alertMessage = "App settings were reset. Onboarding will appear on the next launch."
            refreshResetSizes()
            Task { await refreshRuntime() }
            return
        }
        do {
            let freed = try ResetPlan.remove(scope, libraryRoot: library.baseURL)
            if scope == .library {
                records = []
                selectedRecordID = nil
                createRecordID = nil
                try persist()
            }
            alertMessage = freed > 0
                ? "\(scope.title) removed — \(ResetPlan.formattedSize(freed)) freed."
                : "\(scope.title) removed."
        } catch {
            alertMessage = "Could not remove \(scope.title.lowercased()): \(error.localizedDescription)"
            refreshResetSizes()
            return
        }
        refreshResetSizes()
        Task { await refreshRuntime() }
    }

    /// Full clean uninstall: wipe models, runtime, library, and settings, say
    /// goodbye, then quit. Only the .app bundle remains for the user to trash.
    func resetEverythingAndQuit() {
        stopBackgroundWork()
        for key in ResetPlan.defaultsKeys { UserDefaults.standard.removeObject(forKey: key) }
        do {
            try ResetPlan.removeEverything(libraryRoot: library.baseURL)
        } catch {
            let alert = NSAlert()
            alert.messageText = "FastWan QAD could not be fully removed"
            alert.informativeText = error.localizedDescription
            alert.addButton(withTitle: "OK")
            alert.runModal()
            return
        }
        records = []
        selectedRecordID = nil
        createRecordID = nil
        let alert = NSAlert()
        alert.messageText = "FastWan QAD has been removed"
        alert.informativeText = "All models, the runtime, your library, and settings were deleted from this Mac. Move FastWan QAD.app to the Trash to complete the uninstall."
        alert.addButton(withTitle: "Quit FastWan QAD")
        alert.runModal()
        NSApplication.shared.terminate(nil)
    }

    private func stopBackgroundWork() {
        if isGenerating { cancelGeneration() }
        if isInstallingModel || isInstallingRuntime {
            utilityProcess.cancel()
            isInstallingModel = false
            isInstallingRuntime = false
            installingVariant = nil
        }
    }

    func export(_ record: GenerationRecord) {
        guard let source = record.outputURL, FileManager.default.fileExists(atPath: source.path) else { return }
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.mpeg4Movie]
        panel.nameFieldStringValue = "fastvideo-\(record.id.uuidString.prefix(8)).mp4"
        guard panel.runModal() == .OK, let destination = panel.url else { return }
        do {
            if FileManager.default.fileExists(atPath: destination.path) { try FileManager.default.removeItem(at: destination) }
            try FileManager.default.copyItem(at: source, to: destination)
        } catch {
            alertMessage = "Could not export video: \(error.localizedDescription)"
        }
    }

    func reveal(_ record: GenerationRecord) {
        guard let url = record.outputURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([url])
    }

    func chooseRepositoryRoot() {
        if let path = chooseDirectory(startingAt: configuration.repositoryRoot) {
            configuration.repositoryRoot = path
            let venv = URL(fileURLWithPath: path).appendingPathComponent(".venv/bin/python").path
            if FileManager.default.isExecutableFile(atPath: venv) { configuration.pythonExecutable = venv }
            saveConfiguration()
        }
    }

    func chooseModelRoot() {
        if let path = chooseDirectory(startingAt: configuration.modelRoot) {
            configuration.modelRoot = path
            saveConfiguration()
        }
    }

    func choosePython() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: configuration.pythonExecutable).deletingLastPathComponent()
        if panel.runModal() == .OK, let path = panel.url?.path {
            configuration.pythonExecutable = path
            saveConfiguration()
        }
    }

    private func chooseDirectory(startingAt path: String) -> String? {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: path)
        return panel.runModal() == .OK ? panel.url?.path : nil
    }

    private func handleInstallLine(_ line: String) {
        guard let event = decodeEvent(line) else { setupLog.append(line); return }
        if let message = event.message { setupLog.append(message) }
        if event.type == "progress" { modelInstallProgress = event.fraction }
        if event.type == "complete" { modelInstallProgress = 1 }
        if event.type == "error" { alertMessage = event.message }
    }

    private func handleGenerationLine(_ line: String, id: UUID) {
        guard let event = decodeEvent(line), let index = records.firstIndex(where: { $0.id == id }) else { return }
        if let phase = event.phase { records[index].phase = phase }
        if let fraction = event.fraction { records[index].progress = max(records[index].progress, fraction) }
        switch event.type {
        case "preview":
            if let previewPath = event.previewPath {
                records[index].previewPath = previewPath
            }
        case "complete":
            generationCompletedEvent = true
            records[index].status = .completed
            records[index].progress = 1
            records[index].phase = "Ready"
            records[index].finishedAt = Date()
            if let output = event.outputPath { records[index].outputPath = output }
            records[index].previewPath = nil
            records[index].metrics = GenerationMetrics(dictionary: event.metrics)
            notifyCompletion(record: records[index])
        case "error":
            records[index].status = .failed
            records[index].phase = "Failed"
            records[index].error = event.message ?? "Generation failed."
            records[index].finishedAt = Date()
        default:
            break
        }
        try? persist()
    }

    private func finishGenerationProcess(id: UUID, status: Int32) {
        defer { endGenerationActivity() }
        guard let index = records.firstIndex(where: { $0.id == id }) else { return }
        if records[index].status == .cancelled { return }
        if status != 0 && records[index].status == .running {
            records[index].status = .failed
            records[index].phase = "Failed"
            records[index].finishedAt = Date()
            records[index].error = "The MLX process exited with code \(status)."
        } else if status == 0 && !generationCompletedEvent && records[index].status == .running {
            records[index].status = .failed
            records[index].phase = "Failed"
            records[index].finishedAt = Date()
            records[index].error = "The MLX process ended without a completion event."
        }
        try? persist()
    }

    private func endGenerationActivity() {
        if let activity = generationActivity { ProcessInfo.processInfo.endActivity(activity) }
        generationActivity = nil
    }

    private func persist() throws { try library.save(records) }

    private func decodeEvent(_ line: String) -> BridgeEvent? {
        guard let data = line.data(using: .utf8) else { return nil }
        return try? JSONDecoder().decode(BridgeEvent.self, from: data)
    }

    private func notifyCompletion(record: GenerationRecord) {
        let content = UNMutableNotificationContent()
        content.title = "Your FastWan QAD video is ready"
        content.body = String(record.prompt.prefix(110))
        content.sound = .default
        UNUserNotificationCenter.current().add(UNNotificationRequest(identifier: record.id.uuidString, content: content, trigger: nil))
    }

    private static func runtimeHealth(from object: [String: Any]) -> RuntimeHealth {
        var health = RuntimeHealth()
        health.platformSupported = object["platform_supported"] as? Bool ?? false
        health.mlxAvailable = object["mlx_available"] as? Bool ?? false
        health.torchAvailable = object["torch_available"] as? Bool ?? false
        health.mpsAvailable = object["mps_available"] as? Bool ?? false
        health.rifeAvailable = object["rife_available"] as? Bool ?? false
        health.ffmpegAvailable = object["ffmpeg_available"] as? Bool ?? false
        health.modelComponentsPresent = object["model_components_present"] as? Bool ?? false
        health.rawAvailable = object["raw_available"] as? Bool ?? false
        health.emaAvailable = object["ema_available"] as? Bool ?? false
        health.rawCheckpoint = object["raw_checkpoint"] as? String
        health.emaCheckpoint = object["ema_checkpoint"] as? String
        health.pythonVersion = object["python"] as? String ?? ""
        health.macOSVersion = object["macos"] as? String ?? ""
        health.state = object["ready"] as? Bool == true ? .ready : .needsSetup
        return health
    }
}

private enum AppError: LocalizedError {
    case commandFailed(String)

    var errorDescription: String? {
        switch self {
        case let .commandFailed(output): output.isEmpty ? "The command failed." : output
        }
    }
}
