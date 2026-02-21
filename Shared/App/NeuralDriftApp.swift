import SwiftUI

@main
struct NeuralDriftApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
        #if os(macOS)
        .defaultSize(width: 1280, height: 720)
        #endif
    }
}
