import Foundation

struct ReferenceEntry: Identifiable {
    let id = UUID()
    let displayName: String
    let mapFileName: String        // bundle resource name WITHOUT .json
}

struct TestFileEntry: Identifiable {
    let id = UUID()
    let displayName: String
    let mp3FileName: String        // bundle resource name WITHOUT .mp3
}

struct ReferenceLibrary {

    // Scanned once at first access, then cached
    static let references: [ReferenceEntry] = scanReferences()
    static let testFiles:  [TestFileEntry]  = scanTestFiles()

    // MARK: - Scanning

    /// Find every .json file in the bundle and treat it as a reference map.
    private static func scanReferences() -> [ReferenceEntry] {
        guard let urls = Bundle.main.urls(forResourcesWithExtension: "json",
                                          subdirectory: nil) else {
            print("ReferenceLibrary: no .json files found in bundle")
            return []
        }

        let entries = urls
            .map { url -> ReferenceEntry in
                let stem = url.deletingPathExtension().lastPathComponent
                return ReferenceEntry(displayName: displayName(from: stem),
                                      mapFileName: stem)
            }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }

        if entries.isEmpty {
            print("ReferenceLibrary: no .json files found in bundle")
        }
        return entries
    }

    /// Find every .mp3 file in the bundle.
    private static func scanTestFiles() -> [TestFileEntry] {
        guard let urls = Bundle.main.urls(forResourcesWithExtension: "mp3",
                                          subdirectory: nil) else {
            print("ReferenceLibrary: no .mp3 files found in bundle")
            return []
        }

        let entries = urls
            .map { url -> TestFileEntry in
                let stem = url.deletingPathExtension().lastPathComponent
                return TestFileEntry(displayName: displayName(from: stem),
                                     mp3FileName: stem)
            }
            .sorted { $0.displayName.localizedCaseInsensitiveCompare($1.displayName) == .orderedAscending }

        if entries.isEmpty {
            print("ReferenceLibrary: no .mp3 files found in bundle")
        }
        return entries
    }

    // MARK: - Display-name formatting

    /// Turns a raw filename stem into a human-readable title.
    ///
    /// Examples:
    ///   "jazz_trio"            →  "Jazz Trio"
    ///   "rock_show_demo"       →  "Rock Show Demo"
    ///   "test_performance"     →  "Test Performance"
    ///   "MyBand-Setlist"       →  "Myband Setlist"
    ///   "rehearsal"            →  "Rehearsal"
    private static func displayName(from stem: String) -> String {
        // Replace underscores and hyphens with spaces, then title-case each word
        let words = stem
            .replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst().lowercased() }

        let result = words.joined(separator: " ")
        return result.isEmpty ? stem : result
    }
}
//import Foundation
//
//struct ReferenceEntry: Identifiable {
//    let id = UUID()
//    let displayName: String
//    let mapFileName: String        // e.g. "map_rockshow" (no extension)
//}
//
//struct TestFileEntry: Identifiable {
//    let id = UUID()
//    let displayName: String
//    let mp3FileName: String        // e.g. "test_performance" (no extension)
//}
//
//struct ReferenceLibrary {
//    // ── Add your references here ──────────────────────────────────────────
//    // mapFileName must match the .json file added to your Xcode bundle.
//    static let references: [ReferenceEntry] = [
//        ReferenceEntry(displayName: "Rock Show (Demo)",   mapFileName: "map"),
//        ReferenceEntry(displayName: "Jazz Trio",          mapFileName: "map_jazz"),
//        ReferenceEntry(displayName: "Orchestra",          mapFileName: "map_orchestra"),
//    ]
//
//    // ── Add your test MP3s here ───────────────────────────────────────────
//    // mp3FileName must match the .mp3 file added to your Xcode bundle.
//    static let testFiles: [TestFileEntry] = [
//        TestFileEntry(displayName: "My Test Recording",   mp3FileName: "test_performance"),
//        TestFileEntry(displayName: "Rehearsal Take 2",    mp3FileName: "rehearsal_take2"),
//        TestFileEntry(displayName: "Soundcheck",          mp3FileName: "soundcheck"),
//    ]
//}
