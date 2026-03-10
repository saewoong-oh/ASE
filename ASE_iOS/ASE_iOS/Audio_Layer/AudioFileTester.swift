import Foundation
import AVFoundation
import Combine

// MARK: - JSON model

struct FrequencyMapData: Codable {
    let stem_names: [String]
    let combined_chroma: [[Double]]
    let stems_rms: [String: [Double]]
}

// MARK: - Playback / tracking mode

enum TrackingMode {
    case idle
    case playingFile
    case listeningMic
}

// MARK: - AudioFileTester

class AudioFileTester: ObservableObject {

    // ── Engine ────────────────────────────────────────────────────────────
    private let engineWrapper = ASEWrapper()

    // ── AVAudio objects ───────────────────────────────────────────────────
    private var avEngine        = AVAudioEngine()
    private var playerNode      = AVAudioPlayerNode()
    private var tapInstalled    = false

    // ── Published UI state ────────────────────────────────────────────────
    @Published var refTime:       Double = 0.0
    @Published var totalDuration: Double = 0.0
    @Published var progress:      Double = 0.0
    @Published var confidence:    Double = 0.0
    @Published var overallGainDB: Double = 0.0
    @Published var stemNames:     [String] = []
    @Published var stemGains:     [String: Double] = [:]
    @Published var stemLiveDB:    [String: Double] = [:]
    @Published var stemRefDB:     [String: Double] = [:]
    @Published var isReady:       Bool = false
    @Published var trackingMode:  TrackingMode = .idle

    // Convenience
    var isPlaying:    Bool { trackingMode == .playingFile  }
    var isListening:  Bool { trackingMode == .listeningMic }

    // ── Reference loading ─────────────────────────────────────────────────

    func loadReferenceData(jsonURL: URL) {
        // Stop whatever is running before swapping reference
        stopAll()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let data    = try Data(contentsOf: jsonURL)
                let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)

                let chromaNS = decoded.combined_chroma.map { row in
                    row.map { NSNumber(value: $0) }
                }
                var rmsNS: [String: [NSNumber]] = [:]
                for (key, value) in decoded.stems_rms {
                    rmsNS[key] = value.map { NSNumber(value: $0) }
                }

                self.engineWrapper.loadReferenceChroma(chromaNS,
                                                       stemNames: decoded.stem_names,
                                                       stemRMS: rmsNS)

                DispatchQueue.main.async {
                    self.stemNames    = decoded.stem_names
                    self.totalDuration = Double(decoded.combined_chroma.count)
                                        * (1024.0 / 44100.0)
                    self.isReady      = true
                    print("Reference loaded: \(decoded.stem_names)")
                }
            } catch {
                print("Failed to load reference JSON: \(error)")
            }
        }
    }

    // MARK: - File playback

    func runTest(with audioFileURL: URL) {
        guard isReady else { print("Engine not ready"); return }
        stopAll()
        setupAudioSession(forMic: false)

        do {
            let audioFile  = try AVAudioFile(forReading: audioFileURL)
            let fileFormat = audioFile.processingFormat

            avEngine    = AVAudioEngine()
            playerNode  = AVAudioPlayerNode()
            avEngine.attach(playerNode)

            let mainMixer = avEngine.mainMixerNode
            avEngine.connect(playerNode, to: mainMixer, format: fileFormat)

            installTap(on: mainMixer)

            try avEngine.start()

            playerNode.scheduleFile(audioFile, at: nil) { [weak self] in
                DispatchQueue.main.async { self?.trackingMode = .idle }
                print("File playback finished")
            }
            playerNode.play()

            DispatchQueue.main.async { self.trackingMode = .playingFile }

        } catch {
            print("Failed to start file playback: \(error)")
        }
    }

    // MARK: - Mic tracking

    func startMicTracking() {
        guard isReady else { print("Engine not ready"); return }
        stopAll()
        setupAudioSession(forMic: true)

        avEngine = AVAudioEngine()

        let inputNode  = avEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        print("Mic format: \(inputFormat)")

        // Connect input → mainMixer so the graph is valid, but mute output
        let mainMixer = avEngine.mainMixerNode
        avEngine.connect(inputNode, to: mainMixer, format: inputFormat)
        mainMixer.outputVolume = 0.0          // no speaker feedback

        installTap(on: inputNode, format: inputFormat)

        do {
            try avEngine.start()
            DispatchQueue.main.async { self.trackingMode = .listeningMic }
            print("Mic tracking started")
        } catch {
            print("Failed to start mic tracking: \(error)")
            removeTap()
        }
    }

    // MARK: - Stop everything

    func stopAll() {
        removeTap()
        if avEngine.isRunning {
            if trackingMode == .playingFile { playerNode.stop() }
            avEngine.mainMixerNode.outputVolume = 1.0
            avEngine.stop()
        }
        DispatchQueue.main.async { self.trackingMode = .idle }
    }

    // Convenience aliases kept for backward compatibility
    func stopPlayback() { stopAll() }

    // MARK: - Audio session

    private func setupAudioSession(forMic: Bool) {
        let session = AVAudioSession.sharedInstance()
        do {
            let options: AVAudioSession.CategoryOptions = forMic
                ? [.defaultToSpeaker, .allowBluetooth]
                : [.defaultToSpeaker, .allowBluetooth]

            try session.setCategory(.playAndRecord, mode: .measurement, options: options)
            try session.setActive(true)
            print("Audio session active – forMic:\(forMic)")
        } catch {
            print("Audio session setup failed: \(error)")
        }
    }

    // MARK: - Tap helpers

    private func installTap(on node: AVAudioNode,
                            format: AVAudioFormat? = nil) {
        let tapFormat = format ?? node.outputFormat(forBus: 0)
        node.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buf, _ in
            self?.processTapBuffer(buffer: buf)
        }
        tapInstalled = true
    }

    private func removeTap() {
        guard tapInstalled else { return }
        // Try to remove from whichever node has the tap.
        // Safe to call even if the engine is stopped.
        let engine = avEngine
        if trackingMode == .listeningMic {
            engine.inputNode.removeTap(onBus: 0)
        } else {
            engine.mainMixerNode.removeTap(onBus: 0)
        }
        tapInstalled = false
    }

    // MARK: - DSP

    private func processTapBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else { return }

        let channelCount = Int(buffer.format.channelCount)
        var monoSamples  = [Double](repeating: 0.0, count: frameLength)

        if channelCount >= 2 {
            let ch0 = channelData[0], ch1 = channelData[1]
            for i in 0..<frameLength {
                monoSamples[i] = Double(ch0[i] + ch1[i]) * 0.5
            }
        } else {
            let ch0 = channelData[0]
            for i in 0..<frameLength { monoSamples[i] = Double(ch0[i]) }
        }

        let rawStatus: [AnyHashable: Any] = monoSamples.withUnsafeBufferPointer { ptr in
            self.engineWrapper.processBlock(ptr.baseAddress!, length: frameLength)
        }

        DispatchQueue.main.async { self.unpackStatus(rawStatus) }
    }

    // MARK: - Unpack Obj-C dictionary

    private func unpackStatus(_ status: [AnyHashable: Any]) {
        if let n = status["ref_time"]        as? NSNumber { refTime       = n.doubleValue }
        if let n = status["progress"]        as? NSNumber { progress      = n.doubleValue }
        if let n = status["confidence"]      as? NSNumber { confidence    = n.doubleValue }
        if let n = status["overall_gain_db"] as? NSNumber { overallGainDB = n.doubleValue }

        unpackDict(status["stemGains"], into: &stemGains)
        unpackDict(status["stemLive"],  into: &stemLiveDB)
        unpackDict(status["stemRef"],   into: &stemRefDB)
    }

    private func unpackDict(_ raw: Any?, into target: inout [String: Double]) {
        if let d = raw as? [String: NSNumber] {
            for (k, v) in d { target[k] = v.doubleValue }
        } else if let d = raw as? [String: Any] {
            for (k, v) in d { if let n = v as? NSNumber { target[k] = n.doubleValue } }
        }
    }
}


//
//import Foundation
//import AVFoundation
//import Combine
//
//struct FrequencyMapData: Codable {
//    let stem_names: [String]
//    let combined_chroma: [[Double]]
//    let stems_rms: [String: [Double]]
//}
//
//class AudioFileTester: ObservableObject {
//    private let engineWrapper = ASEWrapper()
//    private var avEngine = AVAudioEngine()
//    private var playerNode = AVAudioPlayerNode()
//    private var mixerNode: AVAudioMixerNode?
//    private var processingTapInstalled = false
//
//    @Published var refTime: Double = 0.0
//    @Published var totalDuration: Double = 0.0
//    @Published var progress: Double = 0.0
//    @Published var confidence: Double = 0.0
//    @Published var overallGainDB: Double = 0.0
//    @Published var stemNames: [String] = []
//
//    @Published var stemGains: [String: Double] = [:]
//    @Published var stemLiveDB: [String: Double] = [:]
//    @Published var stemRefDB: [String: Double] = [:]
//
//    @Published var isReady: Bool = false
//    @Published var isPlaying: Bool = false
//
//    func loadReferenceData(jsonURL: URL) {
//        do {
//            let data = try Data(contentsOf: jsonURL)
//            let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)
//
//            self.stemNames = decoded.stem_names
//            self.totalDuration = Double(decoded.combined_chroma.count) * (1024.0 / 44100.0)
//
//            let chromaNS = decoded.combined_chroma.map { row in
//                row.map { NSNumber(value: $0) }
//            }
//
//            var rmsNS: [String: [NSNumber]] = [:]
//            for (key, value) in decoded.stems_rms {
//                rmsNS[key] = value.map { NSNumber(value: $0) }
//            }
//
//            engineWrapper.loadReferenceChroma(chromaNS, stemNames: decoded.stem_names, stemRMS: rmsNS)
//            DispatchQueue.main.async { self.isReady = true }
//        } catch {
//            print("Failed to load map.json: \(error)")
//        }
//    }
//
//    func runTest(with audioFileURL: URL) {
//        guard isReady else {
//            print("Not ready yet")
//            return
//        }
//
//        // Stop any existing playback
//        stopPlayback()
//
//        // Setup audio session for playback + processing
//        setupAudioSession()
//
//        do {
//            let audioFile = try AVAudioFile(forReading: audioFileURL)
//            let fileFormat = audioFile.processingFormat
//
//            print("Audio file format: \(fileFormat)")
//            print("Sample rate: \(fileFormat.sampleRate), channels: \(fileFormat.channelCount)")
//
//            // Build engine graph
//            avEngine = AVAudioEngine()
//            playerNode = AVAudioPlayerNode()
//
//            avEngine.attach(playerNode)
//
//            // Connect player -> mainMixerNode
//            let mainMixer = avEngine.mainMixerNode
//            avEngine.connect(playerNode, to: mainMixer, format: fileFormat)
//
//            // Install tap on mixer output for our DSP processing
//            // Use 1024 frames to match our block size
//            let tapFormat = mainMixer.outputFormat(forBus: 0)
//            print("Tap format: \(tapFormat)")
//
//            mainMixer.installTap(onBus: 0, bufferSize: 1024, format: tapFormat) { [weak self] buffer, time in
//                self?.processTapBuffer(buffer: buffer)
//            }
//            processingTapInstalled = true
//
//            // Start engine
//            try avEngine.start()
//            print("AVAudioEngine started")
//
//            // Schedule and play the file
//            playerNode.scheduleFile(audioFile, at: nil) { [weak self] in
//                DispatchQueue.main.async {
//                    self?.isPlaying = false
//                    print("Playback finished")
//                }
//            }
//
//            playerNode.play()
//            DispatchQueue.main.async { self.isPlaying = true }
//            print("Player started")
//
//        } catch {
//            print("Failed to setup audio engine: \(error)")
//        }
//    }
//
//    func stopPlayback() {
//        if processingTapInstalled {
//            avEngine.mainMixerNode.removeTap(onBus: 0)
//            processingTapInstalled = false
//        }
//        if avEngine.isRunning {
//            playerNode.stop()
//            avEngine.stop()
//        }
//        DispatchQueue.main.async { self.isPlaying = false }
//    }
//
//    private func setupAudioSession() {
//        let session = AVAudioSession.sharedInstance()
//        do {
//            // playAndRecord allows both output and tap processing
//            try session.setCategory(.playAndRecord,
//                                    mode: .default,
//                                    options: [.defaultToSpeaker, .allowBluetooth])
//            try session.setActive(true)
//            print("Audio session configured: \(session.category)")
//        } catch {
//            print("Audio session setup failed: \(error)")
//        }
//    }
//
//    private func processTapBuffer(buffer: AVAudioPCMBuffer) {
//        guard let channelData = buffer.floatChannelData else { return }
//
//        let frameLength = Int(buffer.frameLength)
//        guard frameLength > 0 else { return }
//
//        // Mix down to mono if stereo
//        let channelCount = Int(buffer.format.channelCount)
//        var monoSamples = [Double](repeating: 0.0, count: frameLength)
//
//        if channelCount >= 2 {
//            let ch0 = channelData[0]
//            let ch1 = channelData[1]
//            for i in 0..<frameLength {
//                monoSamples[i] = Double(ch0[i] + ch1[i]) * 0.5
//            }
//        } else {
//            let ch0 = channelData[0]
//            for i in 0..<frameLength {
//                monoSamples[i] = Double(ch0[i])
//            }
//        }
//
//        // Send to C++ engine
//        let rawStatus: [AnyHashable: Any] = monoSamples.withUnsafeBufferPointer { ptr in
//            return self.engineWrapper.processBlock(ptr.baseAddress!, length: frameLength)
//        }
//
//        DispatchQueue.main.async {
//            self.unpackStatus(rawStatus)
//        }
//    }
//
//    // MARK: - Safe unpacking of the Obj-C dictionary
//    private func unpackStatus(_ status: [AnyHashable: Any]) {
//        if let n = status["ref_time"] as? NSNumber {
//            self.refTime = n.doubleValue
//        }
//        if let n = status["progress"] as? NSNumber {
//            self.progress = n.doubleValue
//        }
//        if let n = status["confidence"] as? NSNumber {
//            self.confidence = n.doubleValue
//        }
//        if let n = status["overall_gain_db"] as? NSNumber {
//            self.overallGainDB = n.doubleValue
//        }
//
//        if let gains = status["stemGains"] as? [String: NSNumber] {
//            for (k, v) in gains { self.stemGains[k] = v.doubleValue }
//        } else if let gains = status["stemGains"] as? [String: Any] {
//            for (k, v) in gains {
//                if let n = v as? NSNumber { self.stemGains[k] = n.doubleValue }
//            }
//        }
//
//        if let live = status["stemLive"] as? [String: NSNumber] {
//            for (k, v) in live { self.stemLiveDB[k] = v.doubleValue }
//        } else if let live = status["stemLive"] as? [String: Any] {
//            for (k, v) in live {
//                if let n = v as? NSNumber { self.stemLiveDB[k] = n.doubleValue }
//            }
//        }
//
//        if let ref = status["stemRef"] as? [String: NSNumber] {
//            for (k, v) in ref { self.stemRefDB[k] = v.doubleValue }
//        } else if let ref = status["stemRef"] as? [String: Any] {
//            for (k, v) in ref {
//                if let n = v as? NSNumber { self.stemRefDB[k] = n.doubleValue }
//            }
//        }
//    }
//}
//
