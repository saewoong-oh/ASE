import SwiftUI

// MARK: - ContentView

struct ContentView: View {

    @StateObject private var tester = AudioFileTester()

    // ── Picker selection ──────────────────────────────────────────────────
    @State private var selectedRefIndex:  Int = 0
    @State private var selectedTestIndex: Int = 0
    @State private var loadedRefIndex:    Int? = nil   // which ref is actually loaded

    // ── Sheet presentation ────────────────────────────────────────────────
    @State private var showRefPicker:  Bool = false
    @State private var showTestPicker: Bool = false

    // Convenience
    private var currentRef:  ReferenceEntry  { ReferenceLibrary.references[selectedRefIndex]  }
    private var currentTest: TestFileEntry   { ReferenceLibrary.testFiles[selectedTestIndex]  }

    // ── Helpers ───────────────────────────────────────────────────────────
    private func formatTime(_ t: Double) -> String {
        let m = Int(t) / 60
        let s = t.truncatingRemainder(dividingBy: 60)
        return String(format: "%02d:%04.1f", m, s)
    }

    private func loadSelectedReference() {
        guard let url = Bundle.main.url(forResource: currentRef.mapFileName,
                                        withExtension: "json") else {
            print("\(currentRef.mapFileName).json not found in bundle")
            return
        }
        tester.loadReferenceData(jsonURL: url)
        loadedRefIndex = selectedRefIndex
    }

    // MARK: - Body

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    headerText

                    referencePickerCard
                    testFilePickerCard

                    if tester.isReady {
                        metricsCard
                        if !tester.stemNames.isEmpty { stemCard }
                        controlButtons
                    } else {
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
        // Reference picker sheet
        .sheet(isPresented: $showRefPicker) {
            pickerSheet(
                title: "Select Reference",
                items: ReferenceLibrary.references,
                selectedIndex: $selectedRefIndex,
                label: { $0.displayName },
                onDismiss: {
                    loadSelectedReference()
                    tester.stopAll()
                }
            )
        }
        // Test file picker sheet
        .sheet(isPresented: $showTestPicker) {
            pickerSheet(
                title: "Select Test File",
                items: ReferenceLibrary.testFiles,
                selectedIndex: $selectedTestIndex,
                label: { $0.displayName },
                onDismiss: { tester.stopAll() }
            )
        }
    }

    // MARK: - Sub-views

    private var headerText: some View {
        Text("Automated Sound Engineer")
            .font(.title2).bold()
    }

    // ── Reference selector card ───────────────────────────────────────────
    private var referencePickerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Reference Map", systemImage: "music.note.list")
                .font(.caption).foregroundColor(.secondary)

            Button {
                tester.stopAll()
                showRefPicker = true
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(currentRef.displayName)
                            .font(.headline)
                        Text(currentRef.mapFileName + ".json")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundColor(.secondary)
                }
                .padding(12)
                .background(Color(UIColor.secondarySystemBackground))
                .cornerRadius(10)
            }
            .buttonStyle(.plain)

            // Status badge
            HStack(spacing: 6) {
                Circle()
                    .fill(tester.isReady
                          ? (loadedRefIndex == selectedRefIndex ? Color.green : Color.orange)
                          : Color.red)
                    .frame(width: 8, height: 8)
                Text(tester.isReady
                     ? (loadedRefIndex == selectedRefIndex ? "Loaded" : "Changed – tap to reload")
                     : "Loading…")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    // ── Test file selector card ───────────────────────────────────────────
    private var testFilePickerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label("Test Audio File", systemImage: "waveform")
                .font(.caption).foregroundColor(.secondary)

            Button {
                tester.stopAll()
                showTestPicker = true
            } label: {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(currentTest.displayName)
                            .font(.headline)
                        Text(currentTest.mp3FileName + ".mp3")
                            .font(.caption2).foregroundColor(.secondary)
                    }
                    Spacer()
                    Image(systemName: "chevron.right")
                        .foregroundColor(.secondary)
                }
                .padding(12)
                .background(Color(UIColor.secondarySystemBackground))
                .cornerRadius(10)
            }
            .buttonStyle(.plain)
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    // ── Metrics card ──────────────────────────────────────────────────────
    private var metricsCard: some View {
        VStack(alignment: .leading, spacing: 12) {

            // Mode badge
            HStack {
                Image(systemName: tester.isListening ? "mic.fill" : "play.fill")
                Text(tester.isListening ? "Live Mic" :
                     (tester.isPlaying ? "File Playback" : "Stopped"))
                    .font(.caption).bold()
                Spacer()
            }
            .foregroundColor(tester.isListening ? .purple :
                             (tester.isPlaying ? .blue : .secondary))

            Divider()

            // Position
            HStack {
                Text("Position: \(formatTime(tester.refTime)) / \(formatTime(tester.totalDuration))")
                Spacer()
                Text("\(Int(tester.progress * 100))%")
            }
            ProgressView(value: min(tester.progress, 1.0)).tint(.blue)

            // Confidence
            HStack {
                Text("Confidence:")
                Spacer()
                Text(tester.confidence > 0.5 ? "● LOCKED" :
                     (tester.confidence > 0.2 ? "◐ SEARCHING" : "○ LOST"))
                    .foregroundColor(tester.confidence > 0.5 ? .green :
                                     (tester.confidence > 0.2 ? .orange : .red))
                    .bold()
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

    // ── Stem table card ───────────────────────────────────────────────────
    private var stemCard: some View {
        VStack(alignment: .leading) {
            HStack {
                Text("STEM").frame(width: 70, alignment: .leading)
                Text("LIVE").frame(width: 65, alignment: .trailing)
                Text("REF").frame(width: 65, alignment: .trailing)
                Text("GAIN").frame(width: 65, alignment: .trailing)
            }
            .font(.caption).foregroundColor(.gray)

            Divider()

            ForEach(tester.stemNames, id: \.self) { name in
                HStack {
                    Text(name.capitalized).bold()
                        .frame(width: 70, alignment: .leading)
                    Text(String(format: "%+.1f", tester.stemLiveDB[name] ?? 0))
                        .frame(width: 65, alignment: .trailing)
                    Text(String(format: "%+.1f", tester.stemRefDB[name] ?? 0))
                        .frame(width: 65, alignment: .trailing)
                    let gain = 20.0 * log10(max(tester.stemGains[name] ?? 1.0, 1e-12))
                    Text(String(format: "%+.1f dB", gain))
                        .frame(width: 65, alignment: .trailing)
                        .foregroundColor(name == "drums" ? .cyan :
                                         (abs(gain) > 4 ? .red : .green))
                }
                .font(.system(.subheadline, design: .monospaced))
                .padding(.vertical, 4)
            }
        }
        .padding()
        .background(Color(UIColor.secondarySystemBackground))
        .cornerRadius(12)
    }

    // ── Control buttons ───────────────────────────────────────────────────
    private var controlButtons: some View {
        VStack(spacing: 12) {

            // ── Row 1: File playback ──────────────────────────────────────
            HStack(spacing: 12) {

                // Play / Restart file
                Button {
                    guard let url = Bundle.main.url(
                        forResource: currentTest.mp3FileName,
                        withExtension: "mp3") else {
                        print("\(currentTest.mp3FileName).mp3 not found")
                        return
                    }
                    tester.runTest(with: url)
                } label: {
                    Label(tester.isPlaying ? "Restart File" : "Play File",
                          systemImage: tester.isPlaying ? "arrow.clockwise" : "play.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(tester.isListening ? .gray : .blue)
                .controlSize(.large)
                .disabled(tester.isListening)

                // Stop (only while playing file)
                if tester.isPlaying {
                    Button {
                        tester.stopAll()
                    } label: {
                        Label("Stop", systemImage: "stop.fill")
                            .frame(maxWidth: .infinity)
                    }
                    .buttonStyle(.bordered)
                    .tint(.red)
                    .controlSize(.large)
                }
            }

            // ── Row 2: Mic tracking ───────────────────────────────────────
            HStack(spacing: 12) {

                Button {
                    if tester.isListening {
                        tester.stopAll()
                    } else {
                        tester.startMicTracking()
                    }
                } label: {
                    Label(tester.isListening ? "Stop Mic" : "Live Mic",
                          systemImage: tester.isListening ? "mic.slash.fill" : "mic.fill")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(tester.isListening ? .red : .purple)
                .controlSize(.large)
                .disabled(tester.isPlaying)
            }
        }
    }

    // ── Loading placeholder ───────────────────────────────────────────────
    private var loadingView: some View {
        VStack(spacing: 12) {
            ProgressView("Loading Reference Data…")
            Text(currentRef.displayName)
                .font(.caption).foregroundColor(.secondary)
        }
        .padding(.vertical, 40)
    }

    // MARK: - Generic picker sheet

    private func pickerSheet<T: Identifiable>(
        title: String,
        items: [T],
        selectedIndex: Binding<Int>,
        label: @escaping (T) -> String,
        onDismiss: @escaping () -> Void
    ) -> some View {
        NavigationView {
            List {
                ForEach(Array(items.enumerated()), id: \.offset) { idx, item in
                    Button {
                        selectedIndex.wrappedValue = idx
                        onDismiss()
                        showRefPicker  = false
                        showTestPicker = false
                    } label: {
                        HStack {
                            Text(label(item))
                                .foregroundColor(.primary)
                            Spacer()
                            if idx == selectedIndex.wrappedValue {
                                Image(systemName: "checkmark")
                                    .foregroundColor(.accentColor)
                            }
                        }
                    }
                }
            }
            .navigationTitle(title)
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") {
                        showRefPicker  = false
                        showTestPicker = false
                    }
                }
            }
        }
    }
}



//
//
//import SwiftUI
//
//struct ContentView: View {
//    @StateObject private var tester = AudioFileTester()
//
//    func formatTime(_ time: Double) -> String {
//        let minutes = Int(time) / 60
//        let seconds = time.truncatingRemainder(dividingBy: 60)
//        return String(format: "%02d:%04.1f", minutes, seconds)
//    }
//
//    var body: some View {
//        ScrollView {
//            VStack(spacing: 20) {
//                Text("Automated Sound Engineer")
//                    .font(.title2).bold()
//
//                if tester.isReady {
//                    // Position and Tracking
//                    VStack(alignment: .leading, spacing: 12) {
//                        HStack {
//                            Text("Position: \(formatTime(tester.refTime)) / \(formatTime(tester.totalDuration))")
//                            Spacer()
//                            Text("\(Int(tester.progress * 100))%")
//                        }
//                        ProgressView(value: min(tester.progress, 1.0))
//                            .tint(.blue)
//
//                        HStack {
//                            Text("Confidence:")
//                            Spacer()
//                            Text(tester.confidence > 0.5 ? "● LOCKED" :
//                                    (tester.confidence > 0.2 ? "◐ SEARCHING" : "○ LOST"))
//                                .foregroundColor(tester.confidence > 0.5 ? .green :
//                                                    (tester.confidence > 0.2 ? .orange : .red))
//                                .bold()
//                        }
//                        ProgressView(value: min(tester.confidence, 1.0))
//                            .tint(tester.confidence > 0.5 ? .green : .orange)
//
//                        Text("Overall Level: \(String(format: "%+.1f dB", tester.overallGainDB))")
//                            .font(.headline)
//                            .padding(.top, 5)
//
//                        // Debug row
//                        Text("confidence raw: \(String(format: "%.4f", tester.confidence))")
//                            .font(.caption)
//                            .foregroundColor(.gray)
//                    }
//                    .padding()
//                    .background(Color(UIColor.secondarySystemBackground))
//                    .cornerRadius(12)
//
//                    // Stem Breakdown
//                    if !tester.stemNames.isEmpty {
//                        VStack(alignment: .leading) {
//                            HStack {
//                                Text("STEM").frame(width: 70, alignment: .leading)
//                                Text("LIVE").frame(width: 65, alignment: .trailing)
//                                Text("REF").frame(width: 65, alignment: .trailing)
//                                Text("GAIN").frame(width: 65, alignment: .trailing)
//                            }.font(.caption).foregroundColor(.gray)
//
//                            Divider()
//
//                            ForEach(tester.stemNames, id: \.self) { name in
//                                HStack {
//                                    Text(name.capitalized).bold()
//                                        .frame(width: 70, alignment: .leading)
//
//                                    Text(String(format: "%+.1f", tester.stemLiveDB[name] ?? 0.0))
//                                        .frame(width: 65, alignment: .trailing)
//
//                                    Text(String(format: "%+.1f", tester.stemRefDB[name] ?? 0.0))
//                                        .frame(width: 65, alignment: .trailing)
//
//                                    let gain = 20.0 * log10(max(tester.stemGains[name] ?? 1.0, 1e-12))
//                                    Text(String(format: "%+.1f dB", gain))
//                                        .frame(width: 65, alignment: .trailing)
//                                        .foregroundColor(name == "drums" ? .cyan :
//                                                            (abs(gain) > 4.0 ? .red : .green))
//                                }
//                                .font(.system(.subheadline, design: .monospaced))
//                                .padding(.vertical, 4)
//                            }
//                        }
//                        .padding()
//                        .background(Color(UIColor.secondarySystemBackground))
//                        .cornerRadius(12)
//                    }
//
//                    Spacer(minLength: 20)
//
//                    // Play / Stop buttons
//                    HStack(spacing: 16) {
//                        Button(action: {
//                            if let url = Bundle.main.url(forResource: "test_performance",
//                                                         withExtension: "mp3") {
//                                tester.runTest(with: url)
//                            } else {
//                                print("test_performance.mp3 not found in bundle")
//                            }
//                        }) {
//                            Label(tester.isPlaying ? "Restart" : "Run File Test",
//                                  systemImage: tester.isPlaying ? "arrow.clockwise" : "play.fill")
//                                .frame(maxWidth: .infinity)
//                        }
//                        .buttonStyle(.borderedProminent)
//                        .controlSize(.large)
//
//                        if tester.isPlaying {
//                            Button(action: {
//                                tester.stopPlayback()
//                            }) {
//                                Label("Stop", systemImage: "stop.fill")
//                                    .frame(maxWidth: .infinity)
//                            }
//                            .buttonStyle(.bordered)
//                            .controlSize(.large)
//                            .tint(.red)
//                        }
//                    }
//
//                } else {
//                    Spacer()
//                    ProgressView("Loading Reference Data...")
//                    Spacer()
//                }
//            }
//            .padding()
//        }
//        .onAppear {
//            if let jsonURL = Bundle.main.url(forResource: "map", withExtension: "json") {
//                tester.loadReferenceData(jsonURL: jsonURL)
//            } else {
//                print("map.json not found in bundle")
//            }
//        }
//    }
//}
//
//
