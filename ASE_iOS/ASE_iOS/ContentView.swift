import SwiftUI

struct ContentView: View {

    @StateObject private var tester = AudioFileTester()

    @State private var selectedRefIndex:  Int  = 0
    @State private var selectedTestIndex: Int  = 0
    @State private var loadedRefIndex:    Int? = nil

    @State private var showRefPicker:  Bool = false
    @State private var showTestPicker: Bool = false

    // Safe accessors — never crash on empty lists
    private var currentRef: ReferenceEntry? {
        let refs = ReferenceLibrary.references
        guard !refs.isEmpty, selectedRefIndex < refs.count else { return nil }
        return refs[selectedRefIndex]
    }

    private var currentTest: TestFileEntry? {
        let tests = ReferenceLibrary.testFiles
        guard !tests.isEmpty, selectedTestIndex < tests.count else { return nil }
        return tests[selectedTestIndex]
    }

    // MARK: - Helpers

    private func formatTime(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = t.truncatingRemainder(dividingBy: 60)
        return String(format: "%02d:%04.1f", m, s)
    }

    private func loadSelectedReference() {
        guard let ref = currentRef else {
            print("ReferenceLibrary.references is empty — add map*.json files to the bundle")
            return
        }
        guard let url = Bundle.main.url(forResource: ref.mapFileName,
                                        withExtension: "json") else {
            print("\(ref.mapFileName).json not found in bundle"); return
        }
        tester.loadReferenceData(jsonURL: url)
        loadedRefIndex = selectedRefIndex
    }

    private func startCurrentFile() {
        guard let test = currentTest else {
            print("ReferenceLibrary.testFiles is empty — add .mp3 files to the bundle")
            return
        }
        guard let url = Bundle.main.url(forResource: test.mp3FileName,
                                        withExtension: "mp3") else {
            print("\(test.mp3FileName).mp3 not found in bundle"); return
        }
        tester.runTest(with: url)
    }

    // MARK: - Body

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    Text("Automated Sound Engineer")
                        .font(.title2).bold()

                    // Show a clear error if no files are bundled
                    if ReferenceLibrary.references.isEmpty {
                        missingFilesWarning(
                            message: "No map*.json reference files found in the app bundle.\nAdd at least one and rebuild.")
                    } else {
                        referencePickerCard
                    }

                    if ReferenceLibrary.testFiles.isEmpty {
                        missingFilesWarning(
                            message: "No .mp3 test files found in the app bundle.\nAdd at least one and rebuild.")
                    } else {
                        testFilePickerCard
                    }

                    if tester.isReady {
                        metricsCard
                        if !tester.stemNames.isEmpty {
                            stemCard
                        }
                        // Only show controls when both lists have something
                        if currentRef != nil && currentTest != nil {
                            controlButtons
                        }
                    } else if currentRef != nil {
                        loadingView
                    }
                }
                .padding()
            }
            .navigationTitle("ASE")
            .navigationBarTitleDisplayMode(.inline)
        }
        .navigationViewStyle(.stack)
        .onAppear { loadSelectedReference() }
        .sheet(isPresented: $showRefPicker)  { refPickerSheet  }
        .sheet(isPresented: $showTestPicker) { testPickerSheet }
    }

    // MARK: - Missing files warning

    private func missingFilesWarning(message: String) -> some View {
        HStack(spacing: 12) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundColor(.orange)
                .font(.title2)
            Text(message)
                .font(.caption)
                .foregroundColor(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    // MARK: - Reference picker card

    private var referencePickerCard: some View {
        // currentRef is guaranteed non-nil here (caller checks)
        let ref = currentRef!
        return VStack(alignment: .leading, spacing: 8) {
            Label("Reference Map", systemImage: "music.note.list")
                .font(.caption).foregroundColor(.secondary)

            Button { showRefPicker = true } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(ref.displayName).font(.headline)
                        Text(ref.mapFileName + ".json")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right").foregroundColor(.secondary)
                }
                .padding(12)
                .background(Color(UIColor.tertiarySystemBackground))
                .cornerRadius(10)
            }
            .buttonStyle(.plain)

            HStack(spacing: 6) {
                Circle()
                    .fill(statusColor)
                    .frame(width: 8, height: 8)
                Text(statusLabel)
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var statusColor: Color {
        if !tester.isReady                    { return .red   }
        if loadedRefIndex == selectedRefIndex { return .green }
        return .orange
    }

    private var statusLabel: String {
        if !tester.isReady                    { return "Loading…"           }
        if loadedRefIndex == selectedRefIndex { return "Loaded"             }
        return "Changed — reload to apply"
    }

    // MARK: - Test file picker card

    private var testFilePickerCard: some View {
        let test = currentTest!   // guaranteed non-nil by caller
        return VStack(alignment: .leading, spacing: 8) {
            Label("Test Audio File", systemImage: "waveform")
                .font(.caption).foregroundColor(.secondary)

            Button { showTestPicker = true } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(test.displayName).font(.headline)
                        Text(test.mp3FileName + ".mp3")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right").foregroundColor(.secondary)
                }
                .padding(12)
                .background(Color(UIColor.tertiarySystemBackground))
                .cornerRadius(10)
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    // MARK: - Metrics card

    private var metricsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 6) {
                Image(systemName: modeBadgeIcon)
                Text(modeBadgeLabel)
                    .font(.caption).bold()
                Spacer()
            }
            .foregroundColor(modeBadgeColor)

            Divider()

            HStack {
                Text("Position: \(formatTime(tester.refTime)) / \(formatTime(tester.totalDuration))")
                Spacer()
                Text("\(Int(tester.progress * 100))%")
            }
            ProgressView(value: min(tester.progress, 1.0)).tint(.blue)

            HStack {
                Text("Confidence:")
                Spacer()
                confidenceBadge
            }
            ProgressView(value: min(tester.confidence, 1.0))
                .tint(tester.confidence > 0.5 ? .green : .orange)

            Text("Overall Level: \(String(format: "%+.1f dB", tester.overallGainDB))")
                .font(.headline).padding(.top, 5)

            Text("confidence raw: \(String(format: "%.4f", tester.confidence))")
                .font(.caption).foregroundColor(.gray)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    private var modeBadgeIcon: String {
        switch tester.trackingMode {
        case .listeningMic: return "mic.fill"
        case .playingFile:  return "play.fill"
        case .idle:         return "stop.fill"
        }
    }

    private var modeBadgeLabel: String {
        switch tester.trackingMode {
        case .listeningMic: return "Live Mic"
        case .playingFile:  return "File Playback"
        case .idle:         return "Stopped"
        }
    }

    private var modeBadgeColor: Color {
        switch tester.trackingMode {
        case .listeningMic: return .purple
        case .playingFile:  return .blue
        case .idle:         return .secondary
        }
    }

    private var confidenceBadge: some View {
        let locked = tester.confidence > 0.5
        let search = tester.confidence > 0.2
        return Text(locked ? "● LOCKED" : (search ? "◐ SEARCHING" : "○ LOST"))
            .foregroundColor(locked ? .green : (search ? .orange : .red))
            .bold()
    }

    // MARK: - Stem table

    private var stemCard: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 0) {
                Text("STEM") .frame(width: 72, alignment: .leading)
                Text("LIVE") .frame(width: 64, alignment: .trailing)
                Text("REF")  .frame(width: 64, alignment: .trailing)
                Text("GAIN") .frame(width: 72, alignment: .trailing)
            }
            .font(.caption)
            .foregroundColor(.gray)
            .padding(.horizontal, 4)
            .padding(.bottom, 6)

            Divider()

            ForEach(tester.stemNames, id: \.self) { name in
                stemRow(name: name)
                    .frame(height: 36)
            }
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
        .fixedSize(horizontal: false, vertical: true)
    }

    private func stemRow(name: String) -> some View {
        let liveDB = tester.stemLiveDB[name] ?? 0.0
        let refDB  = tester.stemRefDB[name]  ?? 0.0
        let gain   = tester.stemGains[name]  ?? 1.0
        let gainDB = 20.0 * log10(max(gain, 1e-12))
        let gainColor: Color = name == "drums" ? .cyan : (abs(gainDB) > 4 ? .red : .green)

        return HStack(spacing: 0) {
            Text(name.capitalized)
                .bold()
                .frame(width: 72, alignment: .leading)
                .lineLimit(1)
                .minimumScaleFactor(0.8)
            Text(String(format: "%+.1f", liveDB))
                .frame(width: 64, alignment: .trailing)
                .monospacedDigit()
            Text(String(format: "%+.1f", refDB))
                .frame(width: 64, alignment: .trailing)
                .monospacedDigit()
            Text(String(format: "%+.1f", gainDB))
                .frame(width: 72, alignment: .trailing)
                .foregroundColor(gainColor)
                .monospacedDigit()
        }
        .font(.system(.subheadline, design: .monospaced))
        .padding(.horizontal, 4)
    }

    // MARK: - Control buttons

    private var controlButtons: some View {
        let mode = tester.trackingMode

        return VStack(spacing: 12) {

            // ── Row 1: File playback ──────────────────────────────────────
            HStack(spacing: 12) {

                Button {
                    startCurrentFile()
                } label: {
                    Label(
                        mode == .playingFile ? "Restart File" : "Play File",
                        systemImage: mode == .playingFile ? "arrow.clockwise" : "play.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)
                .controlSize(.large)
                .disabled(mode == .listeningMic)

                Button {
                    tester.stopAll()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .controlSize(.large)
                .opacity(mode == .playingFile ? 1 : 0)
                .allowsHitTesting(mode == .playingFile)
            }

            // ── Row 2: Mic tracking ───────────────────────────────────────
            HStack(spacing: 12) {

                Button {
                    if mode == .listeningMic { tester.stopAll()          }
                    else                     { tester.startMicTracking() }
                } label: {
                    Label(
                        mode == .listeningMic ? "Stop Mic" : "Live Mic",
                        systemImage: mode == .listeningMic ? "mic.slash.fill" : "mic.fill"
                    )
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(mode == .listeningMic ? .red : .purple)
                .controlSize(.large)
                .disabled(mode == .playingFile)

                Button {
                    tester.stopAll()
                } label: {
                    Label("Stop", systemImage: "stop.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .tint(.red)
                .controlSize(.large)
                .opacity(mode == .listeningMic ? 1 : 0)
                .allowsHitTesting(mode == .listeningMic)
            }
        }
    }

    // MARK: - Loading placeholder

    private var loadingView: some View {
        VStack(spacing: 12) {
            ProgressView("Loading Reference Data…")
            if let ref = currentRef {
                Text(ref.displayName)
                    .font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(.vertical, 40)
    }

    // MARK: - Picker sheets

    private var refPickerSheet: some View {
        NavigationView {
            List {
                ForEach(Array(ReferenceLibrary.references.enumerated()),
                        id: \.offset) { idx, entry in
                    Button {
                        let changed = idx != selectedRefIndex
                        selectedRefIndex = idx
                        showRefPicker    = false
                        if changed {
                            let wasPlaying   = tester.trackingMode == .playingFile
                            let wasListening = tester.trackingMode == .listeningMic
                            loadSelectedReference()
                            if wasPlaying {
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                                    startCurrentFile()
                                }
                            } else if wasListening {
                                DispatchQueue.main.asyncAfter(deadline: .now() + 0.8) {
                                    tester.startMicTracking()
                                }
                            }
                        }
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(entry.displayName).foregroundColor(.primary)
                                Text(entry.mapFileName + ".json")
                                    .font(.caption2).foregroundColor(.secondary)
                            }
                            Spacer()
                            if idx == selectedRefIndex {
                                Image(systemName: "checkmark").foregroundColor(.accentColor)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Select Reference")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showRefPicker = false }
                }
            }
        }
    }

    private var testPickerSheet: some View {
        NavigationView {
            List {
                ForEach(Array(ReferenceLibrary.testFiles.enumerated()),
                        id: \.offset) { idx, entry in
                    Button {
                        let changed = idx != selectedTestIndex
                        selectedTestIndex = idx
                        showTestPicker    = false
                        if changed && tester.trackingMode == .playingFile {
                            startCurrentFile()
                        }
                    } label: {
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(entry.displayName).foregroundColor(.primary)
                                Text(entry.mp3FileName + ".mp3")
                                    .font(.caption2).foregroundColor(.secondary)
                            }
                            Spacer()
                            if idx == selectedTestIndex {
                                Image(systemName: "checkmark").foregroundColor(.accentColor)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Select Test File")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { showTestPicker = false }
                }
            }
        }
    }
}

