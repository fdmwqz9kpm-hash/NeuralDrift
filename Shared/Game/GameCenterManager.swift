import GameKit

/// Manages Game Center authentication and leaderboard submissions.
/// Tracks "mutations" (number of interactions) as the primary score metric.
@MainActor
final class GameCenterManager: ObservableObject {
    @Published var isAuthenticated = false
    @Published var playerDisplayName: String = ""

    // Leaderboard IDs (configure in App Store Connect)
    static let mutationsLeaderboardID = "com.neuraldrift.leaderboard.mutations"
    static let worldsLeaderboardID = "com.neuraldrift.leaderboard.worlds"

    // Stats
    @Published var totalMutations: Int = 0
    @Published var worldsDiscovered: Int = 0

    func authenticate() {
        GKLocalPlayer.local.authenticateHandler = { [weak self] viewController, error in
            Task { @MainActor in
                guard let self else { return }

                if let error {
                    print("[GameCenter] Auth error: \(error.localizedDescription)")
                    self.isAuthenticated = false
                    return
                }

                #if os(macOS)
                if let vc = viewController {
                    // Present login window on macOS
                    if let window = NSApplication.shared.keyWindow {
                        window.contentViewController?.presentAsSheet(vc)
                    }
                    return
                }
                #else
                if let vc = viewController {
                    // Present login on iPadOS
                    if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
                       let rootVC = scene.windows.first?.rootViewController {
                        rootVC.present(vc, animated: true)
                    }
                    return
                }
                #endif

                self.isAuthenticated = GKLocalPlayer.local.isAuthenticated
                self.playerDisplayName = GKLocalPlayer.local.displayName
                print("[GameCenter] Authenticated: \(self.playerDisplayName)")
            }
        }
    }

    func recordMutation() {
        totalMutations += 1
    }

    func recordWorldDiscovered() {
        worldsDiscovered += 1
    }

    func submitScores() {
        guard isAuthenticated else { return }

        Task {
            do {
                try await GKLeaderboard.submitScore(
                    totalMutations,
                    context: 0,
                    player: GKLocalPlayer.local,
                    leaderboardIDs: [Self.mutationsLeaderboardID]
                )
                try await GKLeaderboard.submitScore(
                    worldsDiscovered,
                    context: 0,
                    player: GKLocalPlayer.local,
                    leaderboardIDs: [Self.worldsLeaderboardID]
                )
                print("[GameCenter] Scores submitted: \(totalMutations) mutations, \(worldsDiscovered) worlds")
            } catch {
                print("[GameCenter] Score submission error: \(error.localizedDescription)")
            }
        }
    }

    func showLeaderboard() {
        guard isAuthenticated else { return }

        #if os(macOS)
        GKAccessPoint.shared.trigger(handler: { })
        #else
        if let scene = UIApplication.shared.connectedScenes.first as? UIWindowScene,
           let rootVC = scene.windows.first?.rootViewController {
            let gcVC = GKGameCenterViewController(leaderboardID: Self.mutationsLeaderboardID,
                                                   playerScope: .global,
                                                   timeScope: .allTime)
            gcVC.gameCenterDelegate = GameCenterDismisser.shared
            rootVC.present(gcVC, animated: true)
        }
        #endif
    }
}

#if os(iOS)
/// Helper to dismiss Game Center view controller on iPadOS.
class GameCenterDismisser: NSObject, GKGameCenterControllerDelegate {
    static let shared = GameCenterDismisser()
    func gameCenterViewControllerDidFinish(_ gameCenterViewController: GKGameCenterViewController) {
        gameCenterViewController.dismiss(animated: true)
    }
}
#endif
