import Foundation

struct ReferenceEntry: Identifiable {
    let id = UUID()
    let displayName: String
    let mapFileName: String        // e.g. "map_rockshow" (no extension)
}

struct TestFileEntry: Identifiable {
    let id = UUID()
    let displayName: String
    let mp3FileName: String        // e.g. "test_performance" (no extension)
}

struct ReferenceLibrary {
    // ── Add your references here ──────────────────────────────────────────
    // mapFileName must match the .json file added to your Xcode bundle.
    static let references: [ReferenceEntry] = [
        ReferenceEntry(displayName: "Rock Show (Demo)",   mapFileName: "map"),
        ReferenceEntry(displayName: "Jazz Trio",          mapFileName: "map_jazz"),
        ReferenceEntry(displayName: "Orchestra",          mapFileName: "map_orchestra"),
    ]

    // ── Add your test MP3s here ───────────────────────────────────────────
    // mp3FileName must match the .mp3 file added to your Xcode bundle.
    static let testFiles: [TestFileEntry] = [
        TestFileEntry(displayName: "My Test Recording",   mp3FileName: "test_performance"),
        TestFileEntry(displayName: "Rehearsal Take 2",    mp3FileName: "rehearsal_take2"),
        TestFileEntry(displayName: "Soundcheck",          mp3FileName: "soundcheck"),
    ]
}
