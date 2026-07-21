import SwiftUI

@main
struct FastVideoMacApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup {
            RootView()
                .environmentObject(model)
                .frame(minWidth: 980, minHeight: 720)
        }
        .windowStyle(.hiddenTitleBar)
        .defaultSize(width: 1280, height: 860)
        .commands {
            CommandGroup(after: .newItem) {
                Button("New Generation") {
                    model.startNewGeneration()
                }
                .keyboardShortcut("n", modifiers: .command)
                Button("Generate") { model.generate() }
                    .keyboardShortcut(.return, modifiers: [.command])
                    .disabled(model.isGenerating)
            }
        }
    }
}
