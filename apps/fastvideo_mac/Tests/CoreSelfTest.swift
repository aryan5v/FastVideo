import Foundation

@main
struct CoreSelfTest {
    static func main() async throws {
        try generationLibraryPersistsAndRecoversInterruptedJobs()
        playbackUsesPreviewUntilFinalVideoCompletes()
        try generationSettingsPreserveLegacyRenderMode()
        defaultConfigurationPointsAtBridgeInsideRepository()
        try resetPlanOnlyTouchesAppOwnedData()
        try await processDriverCapturesFastFinalLine()
        print("FastVideo Mac core self-test passed")
    }

    private static func generationLibraryPersistsAndRecoversInterruptedJobs() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = GenerationLibrary(baseURL: root)
        let record = GenerationRecord(
            id: UUID(),
            prompt: "A paper boat crosses a rain puddle.",
            createdAt: Date(timeIntervalSince1970: 123),
            status: .running,
            settings: GenerationSettings(),
            outputPath: nil,
            previewPath: nil,
            phase: "Denoising",
            progress: 0.42
        )

        try library.save([record])
        let loaded = try library.load()
        precondition(loaded.count == 1)
        precondition(loaded[0].status == .failed)
        precondition(loaded[0].phase == "Interrupted")
        precondition(loaded[0].finishedAt != nil)
    }

    private static func playbackUsesPreviewUntilFinalVideoCompletes() {
        var record = GenerationRecord(
            id: UUID(),
            prompt: "Test",
            createdAt: Date(),
            status: .running,
            settings: GenerationSettings(),
            outputPath: "/tmp/final.mp4",
            previewPath: "/tmp/preview.mp4",
            phase: "Preview 1 ready",
            progress: 0.4
        )
        precondition(record.playbackURL?.path == "/tmp/preview.mp4")
        record.status = .completed
        precondition(record.playbackURL?.path == "/tmp/final.mp4")
    }

    private static func generationSettingsPreserveLegacyRenderMode() throws {
        precondition(GenerationSettings().mode == .fast)
        let legacy = Data("{\"variant\":\"ema\",\"width\":832,\"height\":480,\"frames\":81,\"fps\":16,\"seed\":1024,\"dmdSteps\":\"1000,757,522\",\"parallelDecode\":false}".utf8)
        let decoded = try JSONDecoder().decode(GenerationSettings.self, from: legacy)
        precondition(decoded.mode == .full)
    }

    private static func defaultConfigurationPointsAtBridgeInsideRepository() {
        let configuration = RuntimeConfiguration.defaults()
        precondition(configuration.bridgePath.hasSuffix("apps/fastvideo_mac/bridge/fastvideo_mlx_bridge.py"))
        precondition(!configuration.modelRoot.isEmpty)
        precondition(configuration.rawCheckpoint.isEmpty || configuration.rawCheckpoint.hasSuffix("mlx-ckpt-cache-qad-v2/int8"))
        precondition(configuration.emaCheckpoint.isEmpty || configuration.emaCheckpoint.hasSuffix("mlx-ckpt-cache-qad-v2-ema/int8"))
    }

    private static func resetPlanOnlyTouchesAppOwnedData() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library = root.appendingPathComponent("FastVideo/Generations", isDirectory: true)
        let modelsRoot = root.appendingPathComponent("FastWan QAD", isDirectory: true)
        let runtimeRoot = root.appendingPathComponent("FastVideo/Runtime", isDirectory: true)
        let unrelated = root.appendingPathComponent("keep-me.txt")
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: modelsRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: runtimeRoot, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: library, withIntermediateDirectories: true)
        try Data(repeating: 1, count: 512).write(to: modelsRoot.appendingPathComponent("model.bin"))
        try Data(repeating: 2, count: 256).write(to: runtimeRoot.appendingPathComponent("runtime.bin"))
        try Data(repeating: 3, count: 128).write(to: library.appendingPathComponent("video.mp4"))
        try Data("keep".utf8).write(to: unrelated)

        let models = ResetPlan.paths(for: .models, libraryRoot: library, applicationSupportRoot: root)
        precondition(models.count == 1)
        precondition(models[0].lastPathComponent == "FastWan QAD")
        let runtime = ResetPlan.paths(for: .runtime, libraryRoot: library, applicationSupportRoot: root)
        precondition(runtime.count == 1)
        precondition(runtime[0].path.contains("FastVideo/Runtime"))
        precondition(ResetPlan.paths(for: .library, libraryRoot: library, applicationSupportRoot: root) == [library])
        precondition(ResetPlan.paths(for: .settings, libraryRoot: library, applicationSupportRoot: root).isEmpty)
        let full = ResetPlan.fullResetPaths(libraryRoot: library, applicationSupportRoot: root)
        precondition(full.count == 2)
        precondition(full[0].lastPathComponent == "FastWan QAD")
        precondition(full[1].lastPathComponent == "FastVideo")
        precondition(ResetPlan.defaultsKeys.contains("fastvideo.onboarding.completed"))
        precondition(ResetPlan.byteCount(at: URL(fileURLWithPath: "/definitely/not/here")) == 0)

        let removedModels = try ResetPlan.remove(.models, libraryRoot: library, applicationSupportRoot: root)
        precondition(removedModels == 512)
        precondition(!FileManager.default.fileExists(atPath: modelsRoot.path))
        precondition(FileManager.default.fileExists(atPath: runtimeRoot.path))
        precondition(FileManager.default.fileExists(atPath: library.path))
        precondition(FileManager.default.fileExists(atPath: unrelated.path))

        try FileManager.default.createDirectory(at: modelsRoot, withIntermediateDirectories: true)
        try Data(repeating: 4, count: 64).write(to: modelsRoot.appendingPathComponent("model.bin"))
        let removedEverything = try ResetPlan.removeEverything(libraryRoot: library, applicationSupportRoot: root)
        precondition(removedEverything == 448)
        precondition(!FileManager.default.fileExists(atPath: modelsRoot.path))
        precondition(!FileManager.default.fileExists(atPath: runtimeRoot.path))
        precondition(!FileManager.default.fileExists(atPath: library.path))
        precondition(FileManager.default.fileExists(atPath: unrelated.path))
    }

    private static func processDriverCapturesFastFinalLine() async throws {
        let result = try await ProcessDriver.runAndCollect(
            executable: "/usr/bin/printf",
            arguments: ["{\"type\":\"diagnosis\",\"ready\":false}\\n"]
        )
        precondition(result.status == 0)
        precondition(result.output.contains("diagnosis"))
    }
}
