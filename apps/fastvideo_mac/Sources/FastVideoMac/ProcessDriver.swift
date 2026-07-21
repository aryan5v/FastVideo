import Darwin
import Foundation

final class ProcessDriver {
    private let lock = NSLock()
    private var process: Process?
    private var buffer = Data()

    var isRunning: Bool {
        lock.withLock { process?.isRunning == true }
    }

    func start(
        executable: String,
        arguments: [String],
        currentDirectory: String? = nil,
        environment: [String: String]? = nil,
        onLine: @escaping (String) -> Void,
        onTermination: @escaping (Int32) -> Void
    ) throws {
        cancel()
        let process = Process()
        let output = Pipe()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.standardOutput = output
        process.standardError = output
        if let currentDirectory {
            process.currentDirectoryURL = URL(fileURLWithPath: currentDirectory)
        }
        if let environment { process.environment = environment }

        lock.withLock {
            self.process = process
            buffer.removeAll(keepingCapacity: true)
        }

        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            self?.consume(data, onLine: onLine)
        }
        process.terminationHandler = { [weak self] process in
            output.fileHandleForReading.readabilityHandler = nil
            let trailingData = output.fileHandleForReading.readDataToEndOfFile()
            if !trailingData.isEmpty {
                self?.consume(trailingData, onLine: onLine)
            }
            let remainder: String? = if let self {
                self.lock.withLock {
                    guard !self.buffer.isEmpty else { return nil }
                    defer { self.buffer.removeAll() }
                    return String(data: self.buffer, encoding: .utf8)
                }
            } else {
                nil
            }
            if let remainder, !remainder.isEmpty { onLine(remainder) }
            self?.lock.withLock { self?.process = nil }
            onTermination(process.terminationStatus)
        }
        try process.run()
    }

    func cancel() {
        let running = lock.withLock { process }
        guard let running, running.isRunning else { return }
        running.terminate()
        let pid = running.processIdentifier
        DispatchQueue.global().asyncAfter(deadline: .now() + 4) {
            if running.isRunning { Darwin.kill(pid, SIGKILL) }
        }
    }

    private func consume(_ data: Data, onLine: (String) -> Void) {
        let lines: [String] = lock.withLock {
            buffer.append(data)
            var output: [String] = []
            while let newline = buffer.firstIndex(of: 0x0A) {
                let lineData = buffer[..<newline]
                buffer.removeSubrange(...newline)
                if let line = String(data: lineData, encoding: .utf8) {
                    output.append(line)
                }
            }
            return output
        }
        lines.forEach(onLine)
    }

    static func runAndCollect(
        executable: String,
        arguments: [String],
        currentDirectory: String? = nil,
        onLine: @escaping (String) -> Void = { _ in }
    ) async throws -> (status: Int32, output: String) {
        try await withCheckedThrowingContinuation { continuation in
            let driver = ProcessDriver()
            let outputLock = NSLock()
            var collected: [String] = []
            do {
                try driver.start(
                    executable: executable,
                    arguments: arguments,
                    currentDirectory: currentDirectory,
                    onLine: { line in
                        outputLock.withLock { collected.append(line) }
                        onLine(line)
                    },
                    onTermination: { status in
                        // Keep the driver alive until its fast child exits.
                        // Without this capture, a one-line command can finish
                        // after the local driver has deallocated and its pipe
                        // buffer is lost before the continuation resumes.
                        _ = driver.isRunning
                        let text = outputLock.withLock { collected.joined(separator: "\n") }
                        continuation.resume(returning: (status, text))
                    }
                )
            } catch {
                continuation.resume(throwing: error)
            }
        }
    }
}

private extension NSLock {
    func withLock<T>(_ operation: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try operation()
    }
}
