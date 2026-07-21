import Foundation

enum AppSection: String, CaseIterable, Identifiable {
    case create
    case library
    case setup

    var id: String { rawValue }

    var label: String {
        switch self {
        case .create: "Create"
        case .library: "Library"
        case .setup: "Models"
        }
    }

    var symbol: String {
        switch self {
        case .create: "sparkles.rectangle.stack"
        case .library: "film.stack"
        case .setup: "shippingbox"
        }
    }
}

enum ModelVariant: String, Codable, CaseIterable, Identifiable {
    case raw
    case ema

    var id: String { rawValue }
    var label: String { rawValue.uppercased() }

    var detail: String {
        switch self {
        case .raw: "Original distilled checkpoint"
        case .ema: "Smoother motion and the recommended default"
        }
    }

    var displayName: String { self == .ema ? "EMA" : "RAW" }
}

enum GenerationStatus: String, Codable {
    case queued
    case running
    case completed
    case failed
    case cancelled
}

enum GenerationMode: String, Codable, CaseIterable, Identifiable {
    case fast
    case full

    var id: String { rawValue }

    var label: String {
        switch self {
        case .fast: "Fast"
        case .full: "Full"
        }
    }

    var detail: String {
        switch self {
        case .fast: "RIFE 2× · about 2.7× faster"
        case .full: "Generate every frame natively"
        }
    }
}

struct GenerationSettings: Codable, Equatable {
    var variant: ModelVariant = .ema
    var mode: GenerationMode = .fast
    var width = 832
    var height = 480
    var frames = 81
    var fps = 16
    var seed = 1024
    var dmdSteps = "1000,757,522"
    var memoryLimitGiB: Double? = nil
    var parallelDecode = false

    var duration: Double { Double(frames) / Double(fps) }
    var resolutionLabel: String { "\(width) × \(height)" }

    init() {}

    private enum CodingKeys: String, CodingKey {
        case variant, mode, width, height, frames, fps, seed, dmdSteps, memoryLimitGiB, parallelDecode
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        variant = try container.decodeIfPresent(ModelVariant.self, forKey: .variant) ?? .ema
        // Records created before fast mode generated every frame natively.
        mode = try container.decodeIfPresent(GenerationMode.self, forKey: .mode) ?? .full
        width = try container.decodeIfPresent(Int.self, forKey: .width) ?? 832
        height = try container.decodeIfPresent(Int.self, forKey: .height) ?? 480
        frames = try container.decodeIfPresent(Int.self, forKey: .frames) ?? 81
        fps = try container.decodeIfPresent(Int.self, forKey: .fps) ?? 16
        seed = try container.decodeIfPresent(Int.self, forKey: .seed) ?? 1024
        dmdSteps = try container.decodeIfPresent(String.self, forKey: .dmdSteps) ?? "1000,757,522"
        memoryLimitGiB = try container.decodeIfPresent(Double.self, forKey: .memoryLimitGiB)
        parallelDecode = try container.decodeIfPresent(Bool.self, forKey: .parallelDecode) ?? false
    }
}

struct GenerationMetrics: Codable, Equatable {
    var totalSeconds: Double?
    var denoiseSeconds: Double?
    var peakMemoryBytes: Int?

    init(dictionary: [String: JSONValue]?) {
        totalSeconds = dictionary?["total_s"]?.doubleValue
        denoiseSeconds = dictionary?["mlx_denoise_s"]?.doubleValue
        peakMemoryBytes = dictionary?["mlx_denoise_peak_bytes"]?.intValue
    }
}

struct GenerationRecord: Identifiable, Codable, Equatable {
    var id: UUID
    var prompt: String
    var createdAt: Date
    var finishedAt: Date?
    var status: GenerationStatus
    var settings: GenerationSettings
    var outputPath: String?
    var previewPath: String?
    var phase: String
    var progress: Double
    var error: String?
    var metrics: GenerationMetrics?

    var outputURL: URL? {
        guard let outputPath else { return nil }
        return URL(fileURLWithPath: outputPath)
    }

    var playbackURL: URL? {
        if status == .completed { return outputURL }
        guard let previewPath else { return nil }
        return URL(fileURLWithPath: previewPath)
    }
}

enum JSONValue: Codable, Equatable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { self = .array(try container.decode([JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value): try container.encode(value)
        case let .number(value): try container.encode(value)
        case let .bool(value): try container.encode(value)
        case let .object(value): try container.encode(value)
        case let .array(value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    var doubleValue: Double? {
        if case let .number(value) = self { value } else { nil }
    }

    var intValue: Int? { doubleValue.map(Int.init) }
}

struct BridgeEvent: Decodable {
    var type: String
    var phase: String?
    var message: String?
    var level: String?
    var fraction: Double?
    var current: Int?
    var total: Int?
    var outputPath: String?
    var previewPath: String?
    var bytesCompleted: Int?
    var bytesTotal: Int?
    var metrics: [String: JSONValue]?

    enum CodingKeys: String, CodingKey {
        case type, phase, message, level, fraction, current, total, metrics
        case outputPath = "output_path"
        case previewPath = "preview_path"
        case bytesCompleted = "bytes_completed"
        case bytesTotal = "bytes_total"
    }
}

struct RuntimeHealth: Equatable {
    enum State: Equatable {
        case checking
        case ready
        case needsSetup
        case error(String)
    }

    var state: State = .checking
    var platformSupported = false
    var mlxAvailable = false
    var torchAvailable = false
    var mpsAvailable = false
    var rifeAvailable = false
    var ffmpegAvailable = false
    var modelComponentsPresent = false
    var rawAvailable = false
    var emaAvailable = false
    var rawCheckpoint: String?
    var emaCheckpoint: String?
    var pythonVersion = ""
    var macOSVersion = ""

    var canGenerate: Bool { state == .ready }

    func variantAvailable(_ variant: ModelVariant) -> Bool {
        variant == .raw ? rawAvailable : emaAvailable
    }

    func checkpoint(for variant: ModelVariant) -> String? {
        variant == .raw ? rawCheckpoint : emaCheckpoint
    }
}

struct RuntimeConfiguration: Codable, Equatable {
    var repositoryRoot: String
    var pythonExecutable: String
    var uvExecutable: String
    var modelRoot: String
    var rawCheckpoint = ""
    var emaCheckpoint = ""

    var bridgePath: String {
        URL(fileURLWithPath: repositoryRoot)
            .appendingPathComponent("apps/fastvideo_mac/bridge/fastvideo_mlx_bridge.py").path
    }

    var modelCatalogPath: String {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("model-catalog.json"),
           FileManager.default.fileExists(atPath: bundled.path) {
            return bundled.path
        }
        return URL(fileURLWithPath: repositoryRoot)
            .appendingPathComponent("apps/fastvideo_mac/Resources/model-catalog.json").path
    }

    static func defaults() -> RuntimeConfiguration {
        let repositoryRoot = discoverRepositoryRoot()
        let managedPython = managedEnvironmentRoot().appendingPathComponent("bin/python").path
        let venvPython = URL(fileURLWithPath: repositoryRoot).appendingPathComponent(".venv/bin/python").path
        var configuration = RuntimeConfiguration(
            repositoryRoot: repositoryRoot,
            pythonExecutable: FileManager.default.isExecutableFile(atPath: managedPython)
                ? managedPython
                : (FileManager.default.isExecutableFile(atPath: venvPython)
                    ? venvPython
                    : findExecutable(named: "python3") ?? "/usr/bin/python3"),
            uvExecutable: findExecutable(named: "uv") ?? "",
            modelRoot: installBaseRoot().appendingPathComponent("Shared", isDirectory: true).path,
            rawCheckpoint: installBaseRoot().appendingPathComponent("RAW", isDirectory: true).path,
            emaCheckpoint: installBaseRoot().appendingPathComponent("EMA", isDirectory: true).path
        )
        configuration.adoptDetectedLocalArtifacts()
        return configuration
    }

    mutating func prepareInstallDestination(for variant: ModelVariant) -> String {
        let base = Self.installBaseRoot()
        if !Self.modelComponentsPresent(at: modelRoot) {
            modelRoot = base.appendingPathComponent("Shared", isDirectory: true).path
        }
        switch variant {
        case .raw:
            if !Self.mlxCheckpointPresent(at: rawCheckpoint) {
                rawCheckpoint = base.appendingPathComponent("RAW", isDirectory: true).path
            }
            return rawCheckpoint
        case .ema:
            if !Self.mlxCheckpointPresent(at: emaCheckpoint) {
                emaCheckpoint = base.appendingPathComponent("EMA", isDirectory: true).path
            }
            return emaCheckpoint
        }
    }

    mutating func adoptDetectedLocalArtifacts() {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let legacyModelRoots = [
            home.appendingPathComponent("models/qad_int8_ema", isDirectory: true).path,
            home.appendingPathComponent("models/qad_int8", isDirectory: true).path,
        ]
        if !Self.modelComponentsPresent(at: modelRoot) || legacyModelRoots.contains(modelRoot) {
            let modelCandidates = [
                "models/qad_int8_v2_ema",
                "models/qad_int8_v2",
                "models/qad_int8_ema",
                "models/qad_int8",
            ].map { home.appendingPathComponent($0, isDirectory: true).path }
            if let detected = modelCandidates.first(where: Self.modelComponentsPresent) {
                modelRoot = detected
            }
        }

        let legacyRaw = home.appendingPathComponent("mlx-ckpt-cache-qad/int8", isDirectory: true).path
        if !Self.mlxCheckpointPresent(at: rawCheckpoint) || rawCheckpoint == legacyRaw {
            rawCheckpoint = Self.firstMLXCheckpoint(in: [
                home.appendingPathComponent("mlx-ckpt-cache-qad-v2/int8", isDirectory: true).path,
                URL(fileURLWithPath: modelRoot).appendingPathComponent("mlx_dit_raw", isDirectory: true).path,
                legacyRaw,
            ]) ?? ""
        }
        let legacyEMA = home.appendingPathComponent("mlx-ckpt-cache-qad-ema/int8", isDirectory: true).path
        if !Self.mlxCheckpointPresent(at: emaCheckpoint) || emaCheckpoint == legacyEMA {
            emaCheckpoint = Self.firstMLXCheckpoint(in: [
                home.appendingPathComponent("mlx-ckpt-cache-qad-v2-ema/int8", isDirectory: true).path,
                URL(fileURLWithPath: modelRoot).appendingPathComponent("mlx_dit_ema", isDirectory: true).path,
                legacyEMA,
            ]) ?? ""
        }
    }

    mutating func adoptBundledRuntime() {
        guard let resources = Bundle.main.resourceURL else { return }
        let bundledSource = resources.appendingPathComponent("fastvideo-source", isDirectory: true)
        if FileManager.default.fileExists(atPath: bundledSource.appendingPathComponent("pyproject.toml").path) {
            repositoryRoot = bundledSource.path
        }
        let bundledUV = resources.appendingPathComponent("bin/uv")
        if FileManager.default.isExecutableFile(atPath: bundledUV.path) {
            uvExecutable = bundledUV.path
        }
    }

    static func managedEnvironmentRoot() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FastVideo/Runtime/.venv", isDirectory: true)
    }

    static func installBaseRoot() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FastWan QAD/Models/v2", isDirectory: true)
    }

    private static func discoverRepositoryRoot() -> String {
        if let resources = Bundle.main.resourceURL {
            let bundledSource = resources.appendingPathComponent("fastvideo-source", isDirectory: true)
            if FileManager.default.fileExists(atPath: bundledSource.appendingPathComponent("pyproject.toml").path) {
                return bundledSource.path
            }
        }
        if let configured = ProcessInfo.processInfo.environment["FASTVIDEO_REPO_ROOT"],
           FileManager.default.fileExists(atPath: URL(fileURLWithPath: configured).appendingPathComponent("pyproject.toml").path) {
            return configured
        }
        var source = URL(fileURLWithPath: #filePath)
        for _ in 0..<8 {
            source.deleteLastPathComponent()
            if FileManager.default.fileExists(atPath: source.appendingPathComponent("pyproject.toml").path) {
                return source.path
            }
        }
        return FileManager.default.currentDirectoryPath
    }

    private static func findExecutable(named name: String) -> String? {
        if let bundled = Bundle.main.resourceURL?.appendingPathComponent("bin/\(name)"),
           FileManager.default.isExecutableFile(atPath: bundled.path) {
            return bundled.path
        }
        let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
        for directory in path.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(directory)).appendingPathComponent(name).path
            if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
        }
        return nil
    }

    private static func modelComponentsPresent(at path: String) -> Bool {
        guard !path.isEmpty else { return false }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("tokenizer", isDirectory: true).path)
            && FileManager.default.fileExists(atPath: root.appendingPathComponent("text_encoder", isDirectory: true).path)
            && FileManager.default.fileExists(atPath: root.appendingPathComponent("transformer/config.json").path)
    }

    private static func mlxCheckpointPresent(at path: String) -> Bool {
        guard !path.isEmpty else { return false }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        return FileManager.default.fileExists(atPath: root.appendingPathComponent("mlx_dit.json").path)
            && FileManager.default.fileExists(atPath: root.appendingPathComponent("mlx_dit.safetensors").path)
    }

    private static func firstMLXCheckpoint(in candidates: [String]) -> String? {
        candidates.first(where: mlxCheckpointPresent)
    }
}

// MARK: - Uninstall & Reset
//
// Everything FastWan QAD adds to a Mac lives in exactly three places:
// `~/Library/Application Support/FastWan QAD` (models), `~/Library/Application
// Support/FastVideo` (runtime + generations), and two UserDefaults keys.
// ResetScope/ResetPlan describe those boundaries so the app can delete only
// what it owns — nothing else on the system is ever touched.

enum ResetScope: String, CaseIterable, Identifiable {
    case models
    case runtime
    case library
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .models: "Downloaded models"
        case .runtime: "Runtime environment"
        case .library: "Library & history"
        case .settings: "App settings"
        }
    }

    var detail: String {
        switch self {
        case .models: "Shared weights, EMA and RAW checkpoints, and RIFE"
        case .runtime: "Managed Python, MLX, and Metal tooling"
        case .library: "Every generated video, preview, and its history"
        case .settings: "Configuration and onboarding preferences"
        }
    }

    var symbol: String {
        switch self {
        case .models: "shippingbox"
        case .runtime: "cpu"
        case .library: "film.stack"
        case .settings: "gearshape"
        }
    }

    var actionLabel: String { self == .settings ? "Reset" : "Delete" }

    var confirmTitle: String {
        switch self {
        case .models: "Delete all downloaded models?"
        case .runtime: "Delete the runtime environment?"
        case .library: "Delete the entire library?"
        case .settings: "Reset app settings?"
        }
    }

    var confirmDetail: String {
        switch self {
        case .models: "Shared weights, EMA and RAW checkpoints, and RIFE will be permanently deleted. You can download them again later."
        case .runtime: "The managed Python, MLX, and Metal environment will be permanently deleted. It can be reinstalled at any time."
        case .library: "Every generated video and its history will be permanently deleted. This cannot be undone."
        case .settings: "Paths and preferences return to their defaults, and onboarding will appear the next time FastWan QAD opens."
        }
    }
}

enum ResetPlan {
    static let defaultsKeys = [
        "fastvideo.runtime.configuration",
        "fastvideo.onboarding.completed",
    ]

    private static func applicationSupport() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    }

    /// Everything the installer downloads lives under this folder.
    static func appDataRoot(applicationSupportRoot: URL = applicationSupport()) -> URL {
        applicationSupportRoot.appendingPathComponent("FastWan QAD", isDirectory: true)
    }

    /// The managed runtime and the generation library share this folder.
    static func fastvideoDataRoot(applicationSupportRoot: URL = applicationSupport()) -> URL {
        applicationSupportRoot.appendingPathComponent("FastVideo", isDirectory: true)
    }

    static func paths(
        for scope: ResetScope,
        libraryRoot: URL,
        applicationSupportRoot: URL = applicationSupport()
    ) -> [URL] {
        switch scope {
        case .models:
            [appDataRoot(applicationSupportRoot: applicationSupportRoot)]
        case .runtime:
            [fastvideoDataRoot(applicationSupportRoot: applicationSupportRoot).appendingPathComponent("Runtime", isDirectory: true)]
        case .library:
            [libraryRoot]
        case .settings:
            []
        }
    }

    static func fullResetPaths(
        libraryRoot _: URL,
        applicationSupportRoot: URL = applicationSupport()
    ) -> [URL] {
        [
            appDataRoot(applicationSupportRoot: applicationSupportRoot),
            fastvideoDataRoot(applicationSupportRoot: applicationSupportRoot),
        ]
    }

    @discardableResult
    static func remove(
        _ scope: ResetScope,
        libraryRoot: URL,
        applicationSupportRoot: URL = applicationSupport()
    ) throws -> Int64 {
        try removeTargets(paths(
            for: scope,
            libraryRoot: libraryRoot,
            applicationSupportRoot: applicationSupportRoot
        ))
    }

    @discardableResult
    static func removeEverything(
        libraryRoot: URL,
        applicationSupportRoot: URL = applicationSupport()
    ) throws -> Int64 {
        try removeTargets(fullResetPaths(
            libraryRoot: libraryRoot,
            applicationSupportRoot: applicationSupportRoot
        ))
    }

    private static func removeTargets(_ targets: [URL]) throws -> Int64 {
        let manager = FileManager.default
        let existing = targets.filter { manager.fileExists(atPath: $0.path) }
        let freed = existing.reduce(Int64(0)) { $0 + byteCount(at: $1) }
        for target in existing {
            try manager.removeItem(at: target)
            if manager.fileExists(atPath: target.path) {
                throw CocoaError(.fileWriteUnknown)
            }
        }
        return freed
    }

    static func byteCount(at url: URL) -> Int64 {
        guard FileManager.default.fileExists(atPath: url.path) else { return 0 }
        let values = try? url.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
        if values?.isRegularFile == true { return Int64(values?.fileSize ?? 0) }
        var total: Int64 = 0
        guard let enumerator = FileManager.default.enumerator(
            at: url,
            includingPropertiesForKeys: [.fileSizeKey],
            options: [.skipsHiddenFiles]
        ) else { return 0 }
        for case let file as URL in enumerator {
            let size = try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize
            total += Int64(size ?? 0)
        }
        return total
    }

    static func formattedSize(_ bytes: Int64) -> String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}
