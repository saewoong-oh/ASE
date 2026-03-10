import Foundation
import AVFoundation
import Combine

// MARK: - JSON model
struct FrequencyMapData: Codable {
    let stem_names: [String]
    let combined_chroma: [[Double]]
    let stems_rms: [String: [Double]]
}

// MARK: - Tracking mode
enum TrackingMode: Equatable {
    case idle
    case playingFile
    case listeningMic
}

// MARK: - AudioFileTester
class AudioFileTester: ObservableObject {

    // ── Engine ────────────────────────────────────────────────────────────
    private let engineWrapper = ASEWrapper()

    // ── AVAudio ───────────────────────────────────────────────────────────
    private var avEngine:   AVAudioEngine?    = nil
    private var playerNode: AVAudioPlayerNode? = nil
    private var tapNode:    AVAudioNode?       = nil
    private var tapInstalled = false

    // ── Published state ───────────────────────────────────────────────────
    @Published var refTime:        Double = 0.0
    @Published var totalDuration:  Double = 0.0
    @Published var progress:       Double = 0.0
    @Published var confidence:     Double = 0.0
    @Published var overallGainDB:  Double = 0.0
    @Published var stemNames:      [String] = []
    @Published var stemGains:      [String: Double] = [:]
    @Published var stemLiveDB:     [String: Double] = [:]
    @Published var stemRefDB:      [String: Double] = [:]
    @Published var isReady:        Bool = false
    @Published var trackingMode:   TrackingMode = .idle

    var isPlaying:   Bool { trackingMode == .playingFile  }
    var isListening: Bool { trackingMode == .listeningMic }

    // MARK: - Reference loading

    func loadReferenceData(jsonURL: URL) {
        stopAll()
        DispatchQueue.main.async { self.isReady = false }

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let data    = try Data(contentsOf: jsonURL)
                let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)

                let chromaNS = decoded.combined_chroma.map { row in
                    row.map { NSNumber(value: $0) }
                }
                var rmsNS: [String: [NSNumber]] = [:]
                for (k, v) in decoded.stems_rms {
                    rmsNS[k] = v.map { NSNumber(value: $0) }
                }

                self.engineWrapper.loadReferenceChroma(chromaNS,
                                                       stemNames: decoded.stem_names,
                                                       stemRMS: rmsNS)
                DispatchQueue.main.async {
                    self.stemNames     = decoded.stem_names
                    self.totalDuration = Double(decoded.combined_chroma.count)
                                        * (1024.0 / 44100.0)
                    self.refTime       = 0
                    self.progress      = 0
                    self.confidence    = 0
                    self.overallGainDB = 0
                    self.stemGains     = [:]
                    self.stemLiveDB    = [:]
                    self.stemRefDB     = [:]
                    self.isReady       = true
                }
            } catch {
                print("Failed to load reference JSON: \(error)")
            }
        }
    }

    // MARK: - File playback

    func runTest(with audioFileURL: URL) {
        guard isReady else { print("Engine not ready"); return }

        destroyEngine()
        engineWrapper.resetTracker()

        refTime       = 0
        progress      = 0
        confidence    = 0
        overallGainDB = 0

        setupAudioSession(forMic: false)

        do {
            let audioFile  = try AVAudioFile(forReading: audioFileURL)
            let fileFormat = audioFile.processingFormat

            let engine = AVAudioEngine()
            let player = AVAudioPlayerNode()
            engine.attach(player)
            engine.connect(player, to: engine.mainMixerNode, format: fileFormat)

            installTap(on: engine.mainMixerNode, engine: engine)

            try engine.start()

            player.scheduleFile(audioFile, at: nil) { [weak self] in
                DispatchQueue.main.async {
                    guard let self, self.trackingMode == .playingFile else { return }
                    self.trackingMode = .idle
                }
            }
            player.play()

            avEngine   = engine
            playerNode = player

            trackingMode = .playingFile

        } catch {
            print("Failed to start file playback: \(error)")
        }
    }

    // MARK: - Mic tracking

    func startMicTracking() {
        guard isReady else { print("Engine not ready"); return }

        destroyEngine()
        engineWrapper.resetTracker()

        refTime       = 0
        progress      = 0
        confidence    = 0
        overallGainDB = 0

        setupAudioSession(forMic: true)

        let engine    = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        print("Mic input format: \(inputFormat)")
        print("Mic input channel count: \(inputFormat.channelCount)")
        print("Mic input sample rate: \(inputFormat.sampleRate)")

        // ── Connect input → mixer so we can hear what the mic is picking up ──
        // This lets you verify the mic is actually receiving signal.
        // The mixer will play the mic signal through the speaker/headphones.
        // WARNING: if using the built-in speaker without headphones this will
        // cause feedback — use headphones when testing this mode.
        engine.connect(inputNode, to: engine.mainMixerNode, format: inputFormat)
        engine.mainMixerNode.outputVolume = 1.0   // audible monitoring ON

        // Tap the input node directly for the DSP pipeline
        installTap(on: inputNode, engine: engine)

        do {
            try engine.start()
            avEngine     = engine
            trackingMode = .listeningMic
            print("Mic tracking started with monitoring enabled.")
            print("You should now hear the mic input through the speaker/headphones.")
        } catch {
            print("Failed to start mic engine: \(error)")
            removeTap()
        }
    }

    // MARK: - Stop

    func stopAll() {
        destroyEngine()
        trackingMode = .idle
    }

    // MARK: - Engine lifecycle

    private func destroyEngine() {
        removeTap()

        if let engine = avEngine {
            if let player = playerNode, engine.isRunning {
                player.stop()
            }
            if engine.isRunning {
                engine.mainMixerNode.outputVolume = 1.0
                engine.stop()
            }
            if let player = playerNode {
                engine.detach(player)
            }
        }

        avEngine   = nil
        playerNode = nil
    }

    // MARK: - Audio session

    private func setupAudioSession(forMic: Bool) {
        let session = AVAudioSession.sharedInstance()
        do {
            if forMic {
                // Use .playAndRecord so we can both capture AND monitor
                // Use headphones to avoid feedback when monitoring is on
                try session.setCategory(.playAndRecord,
                                        mode: .measurement,
                                        options: [.allowBluetooth,
                                                  .allowBluetoothA2DP])
                // Explicitly prefer the built-in mic
                if let builtInMic = session.availableInputs?.first(where: {
                    $0.portType == .builtInMic
                }) {
                    try session.setPreferredInput(builtInMic)
                    print("Preferred input set to: \(builtInMic.portName)")
                } else {
                    print("WARNING: built-in mic not found, using default input")
                }
            } else {
                try session.setCategory(.playAndRecord,
                                        mode: .measurement,
                                        options: [.defaultToSpeaker,
                                                  .allowBluetooth])
            }
            try session.setActive(true)

            // Log what's actually active
            print("Audio session input: \(session.currentRoute.inputs.map { $0.portName })")
            print("Audio session output: \(session.currentRoute.outputs.map { $0.portName })")

        } catch {
            print("Audio session error: \(error)")
        }
    }

    // MARK: - Tap

    private func installTap(on node: AVAudioNode, engine: AVAudioEngine) {
        let fmt = node.outputFormat(forBus: 0)
        print("Installing tap with format: \(fmt)")
        node.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, time in
            // Log signal level occasionally so we can see if the tap is
            // receiving real signal or silence
            if let data = buf.floatChannelData {
                let frameCount = Int(buf.frameLength)
                var sum: Float = 0
                for i in 0..<frameCount { sum += data[0][i] * data[0][i] }
                let rms = sqrt(sum / Float(frameCount))
                if rms > 0.001 {
                    // Only print when there's meaningful signal
                    print(String(format: "TAP RMS: %.5f", rms))
                }
            }
            self?.processTapBuffer(buffer: buf)
        }
        tapNode      = node
        tapInstalled = true
    }

    private func removeTap() {
        guard tapInstalled, let node = tapNode else { return }
        node.removeTap(onBus: 0)
        tapNode      = nil
        tapInstalled = false
    }

    // MARK: - DSP

    private func processTapBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return }

        let ch   = Int(buffer.format.channelCount)
        var mono = [Double](repeating: 0, count: frameLength)
        if ch >= 2 {
            let c0 = channelData[0], c1 = channelData[1]
            for i in 0..<frameLength { mono[i] = Double(c0[i] + c1[i]) * 0.5 }
        } else {
            let c0 = channelData[0]
            for i in 0..<frameLength { mono[i] = Double(c0[i]) }
        }

        let raw: [AnyHashable: Any] = mono.withUnsafeBufferPointer { ptr in
            self.engineWrapper.processBlock(ptr.baseAddress!, length: frameLength)
        }

        DispatchQueue.main.async { self.unpackStatus(raw) }
    }

    // MARK: - Unpack

    private func unpackStatus(_ s: [AnyHashable: Any]) {
        if let n = s["ref_time"]        as? NSNumber { refTime       = n.doubleValue }
        if let n = s["progress"]        as? NSNumber { progress      = n.doubleValue }
        if let n = s["confidence"]      as? NSNumber { confidence    = n.doubleValue }
        if let n = s["overall_gain_db"] as? NSNumber { overallGainDB = n.doubleValue }
        unpackDict(s["stemGains"], into: &stemGains)
        unpackDict(s["stemLive"],  into: &stemLiveDB)
        unpackDict(s["stemRef"],   into: &stemRefDB)
    }

    private func unpackDict(_ raw: Any?, into target: inout [String: Double]) {
        if let d = raw as? [String: NSNumber] {
            for (k, v) in d { target[k] = v.doubleValue }
        } else if let d = raw as? [String: Any] {
            for (k, v) in d { if let n = v as? NSNumber { target[k] = n.doubleValue } }
        }
    }
}
