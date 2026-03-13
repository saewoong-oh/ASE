import Foundation
import AVFoundation
import Combine

// MARK: - JSON Model

/// Codable model matching the JSON structure exported by analyze_and_export.py.
/// Contains the reference chromagram, stem names, and per-stem RMS profiles
/// needed for real-time position tracking and mix advisory.
struct FrequencyMapData: Codable, Sendable {
    let stem_names: [String]
    let combined_chroma: [[Double]]       // Shape: [n_frames][12] — chroma vectors per analysis frame
    let stems_rms: [String: [Double]]     // Per-stem RMS energy indexed by frame number
}

// MARK: - Tracking Mode

/// Represents the current audio input mode for the tracker.
enum TrackingMode: Equatable {
    case idle             // No audio processing active
    case playingFile      // Playing back an audio file through the engine
    case listeningMic     // Capturing live audio from the microphone
}

// MARK: - AudioFileTester

/// Main controller for real-time audio analysis on iOS.
///
/// Manages loading of reference data (JSON tracking map), audio engine setup
/// for both file playback and microphone capture, and bridges audio buffers
/// to the C++ analysis engine (ASEWrapper) for position tracking and mix advisory.
class AudioFileTester: ObservableObject {

    // ── C++ Engine ────────────────────────────────────────────────────
    
    /// The Objective-C++ wrapper around the C++ PositionTracker and RMSMatcher.
    /// Initialized lazily when audio starts, so it can use the actual hardware sample rate
    /// instead of assuming 44100 Hz.
    private var engineWrapper: ASEWrapper?
    
    /// Decoded JSON reference map held in Swift memory.
    /// Injected into the C++ wrapper when the audio engine starts and the true
    /// hardware sample rate is known.
    private var decodedMap: FrequencyMapData?

    // ── AVAudio Components ────────────────────────────────────────────
    
    private var avEngine:     AVAudioEngine?     = nil
    private var playerNode:   AVAudioPlayerNode? = nil
    private var tapNode:      AVAudioNode?       = nil    // The node we installed the tap on
    private var tapInstalled  = false

    // ── Published UI State ────────────────────────────────────────────
    
    @Published var refTime:        Double = 0.0       // Current estimated reference time (seconds)
    @Published var totalDuration:  Double = 0.0       // Total song duration (seconds)
    @Published var progress:       Double = 0.0       // Normalized position [0, 1]
    @Published var confidence:     Double = 0.0       // Tracker confidence [0, 1]
    @Published var overallGainDB:  Double = 0.0       // Overall level difference (dB)
    @Published var stemNames:      [String] = []      // Names of tracked stems
    @Published var stemGains:      [String: Double] = [:]   // Per-stem gain advisory
    @Published var stemLiveDB:     [String: Double] = [:]   // Per-stem live relative dB
    @Published var stemRefDB:      [String: Double] = [:]   // Per-stem reference relative dB
    @Published var isReady:        Bool = false        // Whether reference data is loaded
    @Published var trackingMode:   TrackingMode = .idle

    // Debug counters for diagnostics
    @Published var tapCallCount:     Int    = 0
    @Published var processCallCount: Int   = 0
    @Published var lastResultKeys:   [String] = []

    /// Convenience accessors for the current tracking state.
    var isPlaying:   Bool { trackingMode == .playingFile  }
    var isListening: Bool { trackingMode == .listeningMic }

    // MARK: - Reference Loading

    /// Load and decode a reference JSON tracking map from a file URL.
    ///
    /// The JSON is parsed on a background thread. Once decoded, the stem names
    /// and total duration are published to the UI. The actual injection into
    /// the C++ engine is deferred until audio starts (so the hardware sample
    /// rate is known).
    func loadReferenceData(jsonURL: URL) {
        stopAll()
        DispatchQueue.main.async { self.isReady = false }

        print("=== loadReferenceData: \(jsonURL.lastPathComponent)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let data    = try Data(contentsOf: jsonURL)
                let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)
                
                // Store decoded map; it will be passed to C++ when audio starts
                self.decodedMap = decoded

                DispatchQueue.main.async {
                    self.stemNames     = decoded.stem_names
                    // Duration = frame_count × (hop_size / sample_rate)
                    self.totalDuration = Double(decoded.combined_chroma.count) * (1024.0 / 44100.0)
                    self.isReady       = true
                    print("=== JSON loaded into memory. Ready to launch audio.")
                }
            } catch {
                print("=== FAILED to load reference JSON: \(error)")
            }
        }
    }

    /// Inject the decoded Swift-side reference map into the C++ engine wrapper.
    /// Converts Swift arrays to NSNumber-based collections for Objective-C++ interop.
    private func injectMapIntoWrapper() {
        guard let map = decodedMap, let wrapper = engineWrapper else { return }
        
        // Convert [[Double]] → [[NSNumber]] for Objective-C bridge
        let chromaNS = map.combined_chroma.map { row in row.map { NSNumber(value: $0) } }
        var rmsNS: [String: [NSNumber]] = [:]
        for (k, v) in map.stems_rms { rmsNS[k] = v.map { NSNumber(value: $0) } }
        
        wrapper.loadReferenceChroma(chromaNS, stemNames: map.stem_names, stemRMS: rmsNS)
        print("=== Map successfully injected into C++ Engine.")
    }

    // MARK: - File Playback

    /// Start position tracking by playing back an audio file through the engine.
    ///
    /// Creates an AVAudioPlayerNode, connects it to the mixer, installs a tap
    /// on the mixer output, and initializes the C++ engine at the hardware's
    /// actual output sample rate.
    func runTest(with audioFileURL: URL) {
        guard isReady else { return }
        destroyEngine()

        setupAudioSession(forMic: false)

        do {
            let audioFile  = try AVAudioFile(forReading: audioFileURL)
            let fileFormat = audioFile.processingFormat
            
            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: fileFormat)

            // Query the actual hardware output sample rate from the mixer node.
            // This may differ from the file's sample rate (e.g., 48000 vs 44100).
            let tapFormat = engine.mainMixerNode.outputFormat(forBus: 0)
            
            // Initialize the C++ engine with the true hardware rate so that
            // FFT bin frequencies and hop timing are correct.
            self.engineWrapper = ASEWrapper(sampleRate: tapFormat.sampleRate)
            self.injectMapIntoWrapper()

            installTap(on: engine.mainMixerNode, engine: engine, format: tapFormat)

            try engine.start()
            player.scheduleFile(audioFile, at: nil) { [weak self] in
                // File finished playing — return to idle state
                DispatchQueue.main.async { self?.trackingMode = .idle }
            }
            player.play()

            avEngine   = engine
            playerNode = player
            trackingMode = .playingFile

        } catch { print("=== FAILED to start file playback: \(error)") }
    }

    // MARK: - Mic Tracking

    /// Start position tracking using live microphone input.
    ///
    /// Configures the audio session for recording, creates an engine with
    /// the input node, and initializes the C++ engine at the microphone's
    /// native sample rate.
    func startMicTracking() {
        guard isReady else { return }
        destroyEngine()
        setupAudioSession(forMic: true)

        let engine      = AVAudioEngine()
        let inputNode   = engine.inputNode
        // Use the mic's native output format to avoid unnecessary sample rate conversion
        let tapFormat = inputNode.outputFormat(forBus: 0)
        
        // Initialize C++ engine at the microphone's actual hardware sample rate
        self.engineWrapper = ASEWrapper(sampleRate: tapFormat.sampleRate)
        self.injectMapIntoWrapper()

        engine.connect(inputNode, to: engine.mainMixerNode, format: tapFormat)
        engine.mainMixerNode.outputVolume = 1.0

        installTap(on: inputNode, engine: engine, format: tapFormat)

        do {
            try engine.start()
            avEngine     = engine
            trackingMode = .listeningMic
        } catch { print("Failed to start mic engine: \(error)") }
    }

    // MARK: - Stop / Lifecycle

    /// Stop all audio processing and return to idle state.
    func stopAll() { destroyEngine(); trackingMode = .idle }

    /// Tear down the audio engine: stop playback, remove taps, release resources.
    private func destroyEngine() {
        removeTap()
        if let engine = avEngine {
            if let player = playerNode, engine.isRunning { player.stop() }
            if engine.isRunning { engine.stop() }
            if let player = playerNode { engine.detach(player) }
        }
        avEngine   = nil
        playerNode = nil
    }

    // MARK: - Audio Session

    /// Configure the AVAudioSession for either file playback or microphone capture.
    ///
    /// For microphone mode: uses .playAndRecord with .measurement mode for flat
    /// frequency response, and forces the built-in mic to avoid Bluetooth latency.
    /// For file mode: uses .defaultToSpeaker so audio plays through the loudspeaker.
    private func setupAudioSession(forMic: Bool) {
        let session = AVAudioSession.sharedInstance()
        do {
            if forMic {
                try session.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetoothHFP])
                // Prefer the built-in microphone over external/Bluetooth inputs
                if let builtInMic = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
                    try session.setPreferredInput(builtInMic)
                }
            } else {
                try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            }
            try session.setActive(true)
        } catch { print("=== Audio session error: \(error)") }
    }

    // MARK: - Tap Management

    /// Install a real-time audio tap on the specified node.
    ///
    /// The tap receives 1024-sample buffers at the node's output format rate
    /// and forwards them to the C++ engine via processTapBuffer().
    private func installTap(on node: AVAudioNode, engine: AVAudioEngine, format: AVAudioFormat) {
        print("=== installTap: \(type(of: node))  Actual Hardware sr=\(format.sampleRate)")

        node.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buf, _ in
            self?.processTapBuffer(buffer: buf)
        }
        tapNode      = node
        tapInstalled = true
    }

    /// Remove the audio tap if one is currently installed.
    private func removeTap() {
        guard tapInstalled, let node = tapNode else { return }
        node.removeTap(onBus: 0)
        tapNode      = nil
        tapInstalled = false
    }

    // MARK: - DSP Bridge

    /// Convert an incoming AVAudioPCMBuffer to mono Float64 and feed it to the C++ engine.
    ///
    /// Handles both mono and stereo input formats. Stereo is downmixed by averaging
    /// the two channels. The resulting mono samples are passed to ASEWrapper.processBlock()
    /// which runs STFT, chroma extraction, position tracking, and RMS matching.
    private func processTapBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData, let wrapper = engineWrapper else { return }
        
        let frameLength = Int(buffer.frameLength)
        let ch   = Int(buffer.format.channelCount)
        var mono = [Double](repeating: 0, count: frameLength)
        
        // Downmix to mono: average left+right for stereo, or pass through for mono
        if ch >= 2 {
            let c0 = channelData[0], c1 = channelData[1]
            for i in 0..<frameLength { mono[i] = Double(c0[i] + c1[i]) * 0.5 }
        } else {
            let c0 = channelData[0]
            for i in 0..<frameLength { mono[i] = Double(c0[i]) }
        }

        // Call into the C++ engine and receive a status dictionary
        let raw: [AnyHashable: Any] = mono.withUnsafeBufferPointer { ptr in
            wrapper.processBlock(ptr.baseAddress!, length: frameLength)
        }

        // Unpack results on the main thread for UI updates
        if !raw.isEmpty { DispatchQueue.main.async { self.unpackStatus(raw) } }
    }

    // MARK: - Result Unpacking

    /// Extract tracking results from the C++ engine's output dictionary and update published properties.
    private func unpackStatus(_ s: [AnyHashable: Any]) {
        if let n = s["ref_time"]        as? NSNumber { refTime       = n.doubleValue }
        if let n = s["progress"]        as? NSNumber { progress      = n.doubleValue }
        if let n = s["confidence"]      as? NSNumber { confidence    = n.doubleValue }
        if let n = s["overall_gain_db"] as? NSNumber { overallGainDB = n.doubleValue }
        unpackDict(s["stemGains"], into: &stemGains)
        unpackDict(s["stemLive"],  into: &stemLiveDB)
        unpackDict(s["stemRef"],   into: &stemRefDB)
    }

    /// Helper to unpack an NSDictionary of NSNumber values into a Swift [String: Double] dict.
    private func unpackDict(_ raw: Any?, into target: inout [String: Double]) {
        if let d = raw as? [String: NSNumber] {
            for (k, v) in d { target[k] = v.doubleValue }
        }
    }
}