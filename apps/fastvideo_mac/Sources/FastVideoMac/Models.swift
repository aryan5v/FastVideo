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
        case .setup: "Setup"
        }
    }

    var symbol: String {
        switch self {
        case .create: "sparkles.rectangle.stack"
        case .library: "film.stack"
        case .setup: "slider.horizontal.3"
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
        case .raw: "Direct distilled weights"
        case .ema: "Averaged training weights"
        }
    }
}

enum GenerationStatus: String, Codable {
    case queued
    case running
    case completed
    case failed
    case cancelled
}

struct GenerationSettings: Codable, Equatable {
    var variant: ModelVariant = .raw
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
    var modelRepository = "FastVideo/FastWan-QAD-INT8-1.3B-Diffusers"
    var modelRevision = ""
    var rawCheckpoint = ""
    var emaCheckpoint = ""

    var bridgePath: String {
        URL(fileURLWithPath: repositoryRoot)
            .appendingPathComponent("apps/fastvideo_mac/bridge/fastvideo_mlx_bridge.py").path
    }

    static func defaults() -> RuntimeConfiguration {
        let repositoryRoot = discoverRepositoryRoot()
        let managedPython = managedEnvironmentRoot().appendingPathComponent("bin/python").path
        let venvPython = URL(fileURLWithPath: repositoryRoot).appendingPathComponent(".venv/bin/python").path
        return RuntimeConfiguration(
            repositoryRoot: repositoryRoot,
            pythonExecutable: FileManager.default.isExecutableFile(atPath: managedPython)
                ? managedPython
                : (FileManager.default.isExecutableFile(atPath: venvPython)
                    ? venvPython
                    : findExecutable(named: "python3") ?? "/usr/bin/python3"),
            uvExecutable: findExecutable(named: "uv") ?? "",
            modelRoot: FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Models/FastWan-QAD-INT8-1.3B").path
        )
    }

    static func managedEnvironmentRoot() -> URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FastVideo/Runtime/.venv", isDirectory: true)
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
        let path = ProcessInfo.processInfo.environment["PATH"] ?? ""
        for directory in path.split(separator: ":") {
            let candidate = URL(fileURLWithPath: String(directory)).appendingPathComponent(name).path
            if FileManager.default.isExecutableFile(atPath: candidate) { return candidate }
        }
        return nil
    }
}
