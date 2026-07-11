import Foundation

final class GenerationLibrary {
    let baseURL: URL
    private let historyURL: URL
    private let encoder: JSONEncoder
    private let decoder: JSONDecoder

    init(baseURL: URL? = nil) {
        let root = baseURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FastVideo", isDirectory: true)
        self.baseURL = root.appendingPathComponent("Generations", isDirectory: true)
        historyURL = self.baseURL.appendingPathComponent("history.json")
        encoder = JSONEncoder()
        decoder = JSONDecoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    func load() throws -> [GenerationRecord] {
        try FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        guard FileManager.default.fileExists(atPath: historyURL.path) else { return [] }
        var records = try decoder.decode([GenerationRecord].self, from: Data(contentsOf: historyURL))
        for index in records.indices where records[index].status == .running || records[index].status == .queued {
            records[index].status = .failed
            records[index].phase = "Interrupted"
            records[index].error = "FastWan QAD closed before this generation finished."
            records[index].finishedAt = Date()
        }
        try save(records)
        return records.sorted { $0.createdAt > $1.createdAt }
    }

    func save(_ records: [GenerationRecord]) throws {
        try FileManager.default.createDirectory(at: baseURL, withIntermediateDirectories: true)
        try encoder.encode(records).write(to: historyURL, options: .atomic)
    }

    func outputURL(for id: UUID) throws -> URL {
        let directory = baseURL.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("video.mp4")
    }

    func requestURL(for id: UUID) throws -> URL {
        let directory = baseURL.appendingPathComponent(id.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory.appendingPathComponent("request.json")
    }

    func delete(_ record: GenerationRecord) throws {
        let directory = baseURL.appendingPathComponent(record.id.uuidString, isDirectory: true)
        if FileManager.default.fileExists(atPath: directory.path) {
            try FileManager.default.removeItem(at: directory)
        }
    }
}
