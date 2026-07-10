import Foundation

@main
struct CoreSelfTest {
    static func main() async throws {
        try generationLibraryPersistsAndRecoversInterruptedJobs()
        playbackUsesPreviewUntilFinalVideoCompletes()
        defaultConfigurationPointsAtBridgeInsideRepository()
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

    private static func defaultConfigurationPointsAtBridgeInsideRepository() {
        let configuration = RuntimeConfiguration.defaults()
        precondition(configuration.bridgePath.hasSuffix("apps/fastvideo_mac/bridge/fastvideo_mlx_bridge.py"))
        precondition(!configuration.modelRoot.isEmpty)
        precondition(configuration.rawCheckpoint.isEmpty || configuration.rawCheckpoint.hasSuffix("mlx-ckpt-cache-qad-v2/int8"))
        precondition(configuration.emaCheckpoint.isEmpty || configuration.emaCheckpoint.hasSuffix("mlx-ckpt-cache-qad-v2-ema/int8"))
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
