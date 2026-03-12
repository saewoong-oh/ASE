import Foundation
import AVFoundation
import Combine

// MARK: - JSON model
struct FrequencyMapData: Codable, Sendable {
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
    // 🌟 THE FIX: Wrapper is an optional variable so we can initialize it with the true hardware rate later
    private var engineWrapper: ASEWrapper?
    
    // We store the decoded JSON map in Swift memory so we can inject it when the wrapper is ready
    private var decodedMap: FrequencyMapData?

    // ── AVAudio ───────────────────────────────────────────────────────────
    private var avEngine:     AVAudioEngine?     = nil
    private var playerNode:   AVAudioPlayerNode? = nil
    private var tapNode:      AVAudioNode?       = nil
    private var tapInstalled  = false

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

    @Published var tapCallCount:     Int    = 0
    @Published var processCallCount: Int   = 0
    @Published var lastResultKeys:   [String] = []

    var isPlaying:   Bool { trackingMode == .playingFile  }
    var isListening: Bool { trackingMode == .listeningMic }

    // MARK: - Reference loading

    func loadReferenceData(jsonURL: URL) {
        stopAll()
        DispatchQueue.main.async { self.isReady = false }

        print("=== loadReferenceData: \(jsonURL.lastPathComponent)")

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }
            do {
                let data    = try Data(contentsOf: jsonURL)
                let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)
                
                // Save the map to Swift memory. We will pass it to C++ when we start the audio.
                self.decodedMap = decoded

                DispatchQueue.main.async {
                    self.stemNames     = decoded.stem_names
                    self.totalDuration = Double(decoded.combined_chroma.count) * (1024.0 / 44100.0)
                    self.isReady       = true
                    print("=== JSON loaded into memory. Ready to launch audio.")
                }
            } catch {
                print("=== FAILED to load reference JSON: \(error)")
            }
        }
    }

    // 🌟 HELPER: Injects the saved Swift JSON map into the C++ wrapper
    private func injectMapIntoWrapper() {
        guard let map = decodedMap, let wrapper = engineWrapper else { return }
        
        let chromaNS = map.combined_chroma.map { row in row.map { NSNumber(value: $0) } }
        var rmsNS: [String: [NSNumber]] = [:]
        for (k, v) in map.stems_rms { rmsNS[k] = v.map { NSNumber(value: $0) } }
        
        wrapper.loadReferenceChroma(chromaNS, stemNames: map.stem_names, stemRMS: rmsNS)
        print("=== Map successfully injected into C++ Engine.")
    }

    // MARK: - File playback

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

            // Get the TRUE hardware output format of the mixer
            let tapFormat = engine.mainMixerNode.outputFormat(forBus: 0)
            
            // 🌟 INITIALIZE C++ ENGINE with the correct hardware rate
            self.engineWrapper = ASEWrapper(sampleRate: tapFormat.sampleRate)
            self.injectMapIntoWrapper()

            installTap(on: engine.mainMixerNode, engine: engine, format: tapFormat)

            try engine.start()
            player.scheduleFile(audioFile, at: nil) { [weak self] in
                DispatchQueue.main.async { self?.trackingMode = .idle }
            }
            player.play()

            avEngine   = engine
            playerNode = player
            trackingMode = .playingFile

        } catch { print("=== FAILED to start file playback: \(error)") }
    }

    // MARK: - Mic tracking

    func startMicTracking() {
        guard isReady else { return }
        destroyEngine()
        setupAudioSession(forMic: true)

        let engine      = AVAudioEngine()
        let inputNode   = engine.inputNode
        let tapFormat = inputNode.outputFormat(forBus: 0)
        
        // 🌟 INITIALIZE C++ ENGINE with the correct hardware mic rate
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

    func stopAll() { destroyEngine(); trackingMode = .idle }

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

    // MARK: - Audio session

    private func setupAudioSession(forMic: Bool) {
        let session = AVAudioSession.sharedInstance()
        do {
            if forMic {
                try session.setCategory(.playAndRecord, mode: .measurement, options: [.allowBluetoothHFP])
                if let builtInMic = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
                    try session.setPreferredInput(builtInMic)
                }
            } else {
                try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            }
            try session.setActive(true)
        } catch { print("=== Audio session error: \(error)") }
    }

    // MARK: - Tap

    private func installTap(on node: AVAudioNode, engine: AVAudioEngine, format: AVAudioFormat) {
        print("=== installTap: \(type(of: node))  Actual Hardware sr=\(format.sampleRate)")

        node.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buf, _ in
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
        guard let channelData = buffer.floatChannelData, let wrapper = engineWrapper else { return }
        
        let frameLength = Int(buffer.frameLength)
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
            wrapper.processBlock(ptr.baseAddress!, length: frameLength)
        }

        if !raw.isEmpty { DispatchQueue.main.async { self.unpackStatus(raw) } }
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
        }
    }
}






//import Foundation
//import AVFoundation
//import Combine
//
//// MARK: - JSON model
//// Nonisolated so it can be decoded off the main actor
//struct FrequencyMapData: Codable, Sendable {
//    let stem_names: [String]
//    let combined_chroma: [[Double]]
//    let stems_rms: [String: [Double]]
//}
//
//// MARK: - Tracking mode
//enum TrackingMode: Equatable {
//    case idle
//    case playingFile
//    case listeningMic
//}
//
//// MARK: - AudioFileTester
//class AudioFileTester: ObservableObject {
//
//    // ── Engine ────────────────────────────────────────────────────────────
//    private let engineWrapper = ASEWrapper()
//
//    // ── AVAudio ───────────────────────────────────────────────────────────
//    private var avEngine:     AVAudioEngine?     = nil
//    private var playerNode:   AVAudioPlayerNode? = nil
//    private var tapNode:      AVAudioNode?       = nil
//    private var tapInstalled  = false
//
//    // ── Published state ───────────────────────────────────────────────────
//    @Published var refTime:        Double = 0.0
//    @Published var totalDuration:  Double = 0.0
//    @Published var progress:       Double = 0.0
//    @Published var confidence:     Double = 0.0
//    @Published var overallGainDB:  Double = 0.0
//    @Published var stemNames:      [String] = []
//    @Published var stemGains:      [String: Double] = [:]
//    @Published var stemLiveDB:     [String: Double] = [:]
//    @Published var stemRefDB:      [String: Double] = [:]
//    @Published var isReady:        Bool = false
//    @Published var trackingMode:   TrackingMode = .idle
//
//    @Published var tapCallCount:     Int    = 0
//    @Published var processCallCount: Int   = 0
//    @Published var lastResultKeys:   [String] = []
//
//    var isPlaying:   Bool { trackingMode == .playingFile  }
//    var isListening: Bool { trackingMode == .listeningMic }
//
//    // MARK: - Reference loading
//
//    func loadReferenceData(jsonURL: URL) {
//        stopAll()
//        DispatchQueue.main.async { self.isReady = false }
//
//        print("=== loadReferenceData: \(jsonURL.lastPathComponent)")
//
//        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
//            guard let self else { return }
//            do {
//                let data    = try Data(contentsOf: jsonURL)
//                print("=== JSON bytes: \(data.count)")
//
//                let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)
//                print("=== Decoded stems: \(decoded.stem_names)  "
//                      + "chroma frames: \(decoded.combined_chroma.count)")
//                for (k, v) in decoded.stems_rms {
//                    print("=== stems_rms[\(k)]: \(v.count) frames  peak=\(v.max() ?? 0)")
//                }
//
//                let chromaNS = decoded.combined_chroma.map { row in
//                    row.map { NSNumber(value: $0) }
//                }
//                var rmsNS: [String: [NSNumber]] = [:]
//                for (k, v) in decoded.stems_rms {
//                    rmsNS[k] = v.map { NSNumber(value: $0) }
//                }
//
//                print("=== Calling loadReferenceChroma …")
//                self.engineWrapper.loadReferenceChroma(chromaNS,
//                                                       stemNames: decoded.stem_names,
//                                                       stemRMS: rmsNS)
//                print("=== loadReferenceChroma returned.")
//
//                DispatchQueue.main.async {
//                    self.stemNames     = decoded.stem_names
//                    self.totalDuration = Double(decoded.combined_chroma.count)
//                                        * (1024.0 / 44100.0)
//                    self.refTime       = 0
//                    self.progress      = 0
//                    self.confidence    = 0
//                    self.overallGainDB = 0
//                    self.stemGains     = [:]
//                    self.stemLiveDB    = [:]
//                    self.stemRefDB     = [:]
//                    self.isReady       = true
//                    print("=== isReady = true  stemNames = \(self.stemNames)")
//                }
//            } catch {
//                print("=== FAILED to load reference JSON: \(error)")
//            }
//        }
//    }
//
//    // MARK: - File playback
//
//    func runTest(with audioFileURL: URL) {
//        print("=== runTest called. isReady=\(isReady)")
//        guard isReady else {
//            print("=== ENGINE NOT READY — aborting")
//            return
//        }
//
//        destroyEngine()
//        engineWrapper.resetTracker()
//        print("=== tracker reset")
//
//        refTime          = 0
//        progress         = 0
//        confidence       = 0
//        overallGainDB    = 0
//        tapCallCount     = 0
//        processCallCount = 0
//        lastResultKeys   = []
//
//        setupAudioSession(forMic: false)
//
//        do {
//            let audioFile  = try AVAudioFile(forReading: audioFileURL)
//            let fileFormat = audioFile.processingFormat
//            print("=== AVAudioFile format: \(fileFormat)")
//            print(String(format: "=== File: %lld frames @ %.0f Hz = %.1f s",
//                         audioFile.length,
//                         fileFormat.sampleRate,
//                         Double(audioFile.length) / fileFormat.sampleRate))
//
//            let engine = AVAudioEngine()
//            let player = AVAudioPlayerNode()
//            engine.attach(player)
//            engine.connect(player, to: engine.mainMixerNode, format: fileFormat)
//
//            print("=== Installing tap on mainMixerNode …")
//            installTap(on: engine.mainMixerNode, engine: engine)
//
//            try engine.start()
//            print("=== Engine started.")
//
//            player.scheduleFile(audioFile, at: nil) { [weak self] in
//                DispatchQueue.main.async {
//                    guard let self, self.trackingMode == .playingFile else { return }
//                    print("=== Playback finished.")
//                    self.trackingMode = .idle
//                }
//            }
//            player.play()
//            print("=== Player playing.")
//
//            avEngine   = engine
//            playerNode = player
//            trackingMode = .playingFile
//
//        } catch {
//            print("=== FAILED to start file playback: \(error)")
//        }
//    }
//
//    // MARK: - Mic tracking
//
//    func startMicTracking() {
//        guard isReady else { print("Engine not ready"); return }
//
//        destroyEngine()
//        engineWrapper.resetTracker()
//
//        refTime          = 0
//        progress         = 0
//        confidence       = 0
//        overallGainDB    = 0
//        tapCallCount     = 0
//        processCallCount = 0
//        lastResultKeys   = []
//
//        setupAudioSession(forMic: true)
//
//        let engine      = AVAudioEngine()
//        let inputNode   = engine.inputNode
//        let inputFormat = inputNode.outputFormat(forBus: 0)
//        print("Mic input format: \(inputFormat)")
//
//        engine.connect(inputNode, to: engine.mainMixerNode, format: inputFormat)
//        engine.mainMixerNode.outputVolume = 1.0
//
//        installTap(on: inputNode, engine: engine)
//
//        do {
//            try engine.start()
//            avEngine     = engine
//            trackingMode = .listeningMic
//            print("Mic tracking started.")
//        } catch {
//            print("Failed to start mic engine: \(error)")
//            removeTap()
//        }
//    }
//
//    // MARK: - Stop
//
//    func stopAll() {
//        print("=== stopAll — tapCalls=\(tapCallCount) processCalls=\(processCallCount)")
//        destroyEngine()
//        trackingMode = .idle
//    }
//
//    // MARK: - Engine lifecycle
//
//    private func destroyEngine() {
//        removeTap()
//        if let engine = avEngine {
//            if let player = playerNode, engine.isRunning { player.stop() }
//            if engine.isRunning {
//                engine.mainMixerNode.outputVolume = 1.0
//                engine.stop()
//            }
//            if let player = playerNode { engine.detach(player) }
//        }
//        avEngine   = nil
//        playerNode = nil
//    }
//
//    // MARK: - Audio session
//
//    private func setupAudioSession(forMic: Bool) {
//        let session = AVAudioSession.sharedInstance()
//        do {
//            if forMic {
//                try session.setCategory(.playAndRecord,
//                                        mode: .measurement,
//                                        options: [.allowBluetoothHFP,
//                                                  .allowBluetoothA2DP])
//                if let builtInMic = session.availableInputs?.first(where: {
//                    $0.portType == .builtInMic
//                }) {
//                    try session.setPreferredInput(builtInMic)
//                }
//            } else {
//                try session.setCategory(.playAndRecord,
//                                        mode: .measurement,
//                                        options: [.defaultToSpeaker,
//                                                  .allowBluetoothHFP])
//            }
//            try session.setActive(true)
//            print("=== Audio session: "
//                  + "in=\(session.currentRoute.inputs.map{$0.portName}) "
//                  + "out=\(session.currentRoute.outputs.map{$0.portName})")
//        } catch {
//            print("=== Audio session error: \(error)")
//        }
//    }
//
//    // MARK: - Tap
//
//    private func installTap(on node: AVAudioNode, engine: AVAudioEngine) {
//        let sourceFormat = node.outputFormat(forBus: 0)
//        
//        // 🌟 THE FIX: Force AVAudioEngine to hardware-resample the tap to 44.1kHz
//        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32,
//                                               sampleRate: 44100.0,
//                                               channels: sourceFormat.channelCount,
//                                               interleaved: false) else {
//            print("=== Failed to create 44.1kHz target format")
//            return
//        }
//
//        print("=== installTap: \(type(of: node))  Forcing sr=44100.0  ch=\(sourceFormat.channelCount)")
//
//        node.installTap(onBus: 0, bufferSize: 1024, format: targetFormat) { [weak self] buf, _ in
//            guard let self else { return }
//
//            if let data = buf.floatChannelData {
//                let n = Int(buf.frameLength)
//                var sum: Float = 0
//                for i in 0..<n { sum += data[0][i] * data[0][i] }
//                let rms = sqrt(sum / Float(n))
//                if rms > 0.001 {
//                    // print(String(format: "TAP RMS: %.5f", rms))
//                }
//            }
//
//            let callNum = self.tapCallCount + 1
//            DispatchQueue.main.async { self.tapCallCount = callNum }
//            self.processTapBuffer(buffer: buf, tapCallNumber: callNum)
//        }
//        
//        tapNode      = node
//        tapInstalled = true
//        print("=== Tap installed and locked to 44.1kHz.")
//    }
//
//    private func removeTap() {
//        guard tapInstalled, let node = tapNode else { return }
//        node.removeTap(onBus: 0)
//        tapNode      = nil
//        tapInstalled = false
//        print("=== Tap removed.")
//    }
//
//    // MARK: - DSP
//
//    private func processTapBuffer(buffer: AVAudioPCMBuffer, tapCallNumber: Int) {
//        guard let channelData = buffer.floatChannelData else {
//            print("=== processTapBuffer #\(tapCallNumber): no channelData")
//            return
//        }
//        let frameLength = Int(buffer.frameLength)
//        guard frameLength > 0 else {
//            print("=== processTapBuffer #\(tapCallNumber): frameLength=0")
//            return
//        }
//
//        let ch   = Int(buffer.format.channelCount)
//        var mono = [Double](repeating: 0, count: frameLength)
//        if ch >= 2 {
//            let c0 = channelData[0], c1 = channelData[1]
//            for i in 0..<frameLength { mono[i] = Double(c0[i] + c1[i]) * 0.5 }
//        } else {
//            let c0 = channelData[0]
//            for i in 0..<frameLength { mono[i] = Double(c0[i]) }
//        }
//
//        if tapCallNumber <= 5 {
//            let monoRMS = sqrt(mono.map{$0*$0}.reduce(0,+) / Double(mono.count))
//            print(String(format: "=== processTapBuffer #%d  frames=%d  ch=%d  monoRMS=%.6f",
//                         tapCallNumber, frameLength, ch, monoRMS))
//        }
//
//        let raw: [AnyHashable: Any] = mono.withUnsafeBufferPointer { ptr in
//            self.engineWrapper.processBlock(ptr.baseAddress!, length: frameLength)
//        }
//
//        if tapCallNumber <= 10 {
//            print("=== processBlock returned \(raw.count) keys: \(raw.keys.map{"\($0)"})")
//            if let conf = raw["confidence"] as? NSNumber {
//                print(String(format: "    confidence=%.4f", conf.doubleValue))
//            }
//            if let pos = raw["position"] as? NSNumber {
//                print("    position=\(pos.intValue)")
//            }
//        }
//
//        if !raw.isEmpty {
//            DispatchQueue.main.async {
//                self.processCallCount += 1
//                self.lastResultKeys = raw.keys.compactMap { $0 as? String }
//            }
//        }
//
//        DispatchQueue.main.async { self.unpackStatus(raw) }
//    }
//
//    // MARK: - Unpack
//
//    private func unpackStatus(_ s: [AnyHashable: Any]) {
//        if let n = s["ref_time"]        as? NSNumber { refTime       = n.doubleValue }
//        if let n = s["progress"]        as? NSNumber { progress      = n.doubleValue }
//        if let n = s["confidence"]      as? NSNumber { confidence    = n.doubleValue }
//        if let n = s["overall_gain_db"] as? NSNumber { overallGainDB = n.doubleValue }
//        unpackDict(s["stemGains"], into: &stemGains)
//        unpackDict(s["stemLive"],  into: &stemLiveDB)
//        unpackDict(s["stemRef"],   into: &stemRefDB)
//    }
//
//    private func unpackDict(_ raw: Any?, into target: inout [String: Double]) {
//        if let d = raw as? [String: NSNumber] {
//            for (k, v) in d { target[k] = v.doubleValue }
//        } else if let d = raw as? [String: Any] {
//            for (k, v) in d { if let n = v as? NSNumber { target[k] = n.doubleValue } }
//        }
//    }
//}











//import Foundation
//import AVFoundation
//import Combine
//
//// MARK: - JSON model
//// Nonisolated so it can be decoded off the main actor
//struct FrequencyMapData: Codable, Sendable {
//    let stem_names: [String]
//    let combined_chroma: [[Double]]
//    let stems_rms: [String: [Double]]
//}
//
//// MARK: - Tracking mode
//enum TrackingMode: Equatable {
//    case idle
//    case playingFile
//    case listeningMic
//}
//
//// MARK: - AudioFileTester
//class AudioFileTester: ObservableObject {
//
//    // ── Engine ────────────────────────────────────────────────────────────
//    private let engineWrapper = ASEWrapper()
//
//    // ── AVAudio ───────────────────────────────────────────────────────────
//    private var avEngine:   AVAudioEngine?     = nil
//    private var playerNode: AVAudioPlayerNode? = nil
//    private var tapNode:    AVAudioNode?        = nil
//    private var tapInstalled = false
//
//    // ── Published state ───────────────────────────────────────────────────
//    @Published var refTime:        Double = 0.0
//    @Published var totalDuration:  Double = 0.0
//    @Published var progress:       Double = 0.0
//    @Published var confidence:     Double = 0.0
//    @Published var overallGainDB:  Double = 0.0
//    @Published var stemNames:      [String] = []
//    @Published var stemGains:      [String: Double] = [:]
//    @Published var stemLiveDB:     [String: Double] = [:]
//    @Published var stemRefDB:      [String: Double] = [:]
//    @Published var isReady:        Bool = false
//    @Published var trackingMode:   TrackingMode = .idle
//
//    @Published var tapCallCount:    Int    = 0
//    @Published var processCallCount: Int   = 0
//    @Published var lastResultKeys:  [String] = []
//
//    var isPlaying:   Bool { trackingMode == .playingFile  }
//    var isListening: Bool { trackingMode == .listeningMic }
//
//    // MARK: - Reference loading
//
//    func loadReferenceData(jsonURL: URL) {
//        stopAll()
//        DispatchQueue.main.async { self.isReady = false }
//
//        print("=== loadReferenceData: \(jsonURL.lastPathComponent)")
//
//        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
//            guard let self else { return }
//            do {
//                let data    = try Data(contentsOf: jsonURL)
//                print("=== JSON bytes: \(data.count)")
//
//                let decoded = try JSONDecoder().decode(FrequencyMapData.self, from: data)
//                print("=== Decoded stems: \(decoded.stem_names)  "
//                      + "chroma frames: \(decoded.combined_chroma.count)")
//                for (k, v) in decoded.stems_rms {
//                    print("=== stems_rms[\(k)]: \(v.count) frames  peak=\(v.max() ?? 0)")
//                }
//
//                let chromaNS = decoded.combined_chroma.map { row in
//                    row.map { NSNumber(value: $0) }
//                }
//                var rmsNS: [String: [NSNumber]] = [:]
//                for (k, v) in decoded.stems_rms {
//                    rmsNS[k] = v.map { NSNumber(value: $0) }
//                }
//
//                print("=== Calling loadReferenceChroma …")
//                self.engineWrapper.loadReferenceChroma(chromaNS,
//                                                       stemNames: decoded.stem_names,
//                                                       stemRMS: rmsNS)
//                print("=== loadReferenceChroma returned.")
//
//                DispatchQueue.main.async {
//                    self.stemNames     = decoded.stem_names
//                    self.totalDuration = Double(decoded.combined_chroma.count)
//                                        * (1024.0 / 44100.0)
//                    self.refTime       = 0
//                    self.progress      = 0
//                    self.confidence    = 0
//                    self.overallGainDB = 0
//                    self.stemGains     = [:]
//                    self.stemLiveDB    = [:]
//                    self.stemRefDB     = [:]
//                    self.isReady       = true
//                    print("=== isReady = true  stemNames = \(self.stemNames)")
//                }
//            } catch {
//                print("=== FAILED to load reference JSON: \(error)")
//            }
//        }
//    }
//
//    // MARK: - File playback
//
//    func runTest(with audioFileURL: URL) {
//        print("=== runTest called. isReady=\(isReady)")
//        guard isReady else {
//            print("=== ENGINE NOT READY — aborting")
//            return
//        }
//
//        destroyEngine()
//        engineWrapper.resetTracker()
//        print("=== tracker reset")
//
//        refTime          = 0
//        progress         = 0
//        confidence       = 0
//        overallGainDB    = 0
//        tapCallCount     = 0
//        processCallCount = 0
//        lastResultKeys   = []
//
//        setupAudioSession(forMic: false)
//
//        do {
//            let audioFile  = try AVAudioFile(forReading: audioFileURL)
//            let fileFormat = audioFile.processingFormat
//            print("=== AVAudioFile format: \(fileFormat)")
//            print(String(format: "=== File: %lld frames @ %.0f Hz = %.1f s",
//                         audioFile.length,
//                         fileFormat.sampleRate,
//                         Double(audioFile.length) / fileFormat.sampleRate))
//
//            let engine = AVAudioEngine()
//            let player = AVAudioPlayerNode()
//            engine.attach(player)
//            engine.connect(player, to: engine.mainMixerNode, format: fileFormat)
//
//            print("=== Installing tap on mainMixerNode …")
//            installTap(on: engine.mainMixerNode, engine: engine)
//
//            try engine.start()
//            print("=== Engine started.")
//
//            player.scheduleFile(audioFile, at: nil) { [weak self] in
//                DispatchQueue.main.async {
//                    guard let self, self.trackingMode == .playingFile else { return }
//                    print("=== Playback finished.")
//                    self.trackingMode = .idle
//                }
//            }
//            player.play()
//            print("=== Player playing.")
//
//            avEngine   = engine
//            playerNode = player
//            trackingMode = .playingFile
//
//        } catch {
//            print("=== FAILED to start file playback: \(error)")
//        }
//    }
//
//    // MARK: - Mic tracking
//
//    func startMicTracking() {
//        guard isReady else { print("Engine not ready"); return }
//
//        destroyEngine()
//        engineWrapper.resetTracker()
//
//        refTime          = 0
//        progress         = 0
//        confidence       = 0
//        overallGainDB    = 0
//        tapCallCount     = 0
//        processCallCount = 0
//        lastResultKeys   = []
//
//        setupAudioSession(forMic: true)
//
//        let engine      = AVAudioEngine()
//        let inputNode   = engine.inputNode
//        let inputFormat = inputNode.outputFormat(forBus: 0)
//        print("Mic input format: \(inputFormat)")
//
//        engine.connect(inputNode, to: engine.mainMixerNode, format: inputFormat)
//        engine.mainMixerNode.outputVolume = 1.0
//
//        installTap(on: inputNode, engine: engine)
//
//        do {
//            try engine.start()
//            avEngine     = engine
//            trackingMode = .listeningMic
//            print("Mic tracking started.")
//        } catch {
//            print("Failed to start mic engine: \(error)")
//            removeTap()
//        }
//    }
//
//    // MARK: - Stop
//
//    func stopAll() {
//        print("=== stopAll — tapCalls=\(tapCallCount) processCalls=\(processCallCount)")
//        destroyEngine()
//        trackingMode = .idle
//    }
//
//    // MARK: - Engine lifecycle
//
//    private func destroyEngine() {
//        removeTap()
//        if let engine = avEngine {
//            if let player = playerNode, engine.isRunning { player.stop() }
//            if engine.isRunning {
//                engine.mainMixerNode.outputVolume = 1.0
//                engine.stop()
//            }
//            if let player = playerNode { engine.detach(player) }
//        }
//        avEngine   = nil
//        playerNode = nil
//    }
//
//    // MARK: - Audio session
//
//    private func setupAudioSession(forMic: Bool) {
//        let session = AVAudioSession.sharedInstance()
//        do {
//            if forMic {
//                try session.setCategory(.playAndRecord,
//                                        mode: .measurement,
//                                        options: [.allowBluetoothHFP,
//                                                  .allowBluetoothA2DP])
//                if let builtInMic = session.availableInputs?.first(where: {
//                    $0.portType == .builtInMic
//                }) {
//                    try session.setPreferredInput(builtInMic)
//                }
//            } else {
//                try session.setCategory(.playAndRecord,
//                                        mode: .measurement,
//                                        options: [.defaultToSpeaker,
//                                                  .allowBluetoothHFP])
//            }
//            try session.setActive(true)
//            print("=== Audio session: "
//                  + "in=\(session.currentRoute.inputs.map{$0.portName}) "
//                  + "out=\(session.currentRoute.outputs.map{$0.portName})")
//        } catch {
//            print("=== Audio session error: \(error)")
//        }
//    }
//
//    // MARK: - Tap
//
//    private func installTap(on node: AVAudioNode, engine: AVAudioEngine) {
//        let fmt = node.outputFormat(forBus: 0)
//        print("=== installTap: \(type(of: node))  sr=\(fmt.sampleRate)  ch=\(fmt.channelCount)")
//
//        node.installTap(onBus: 0, bufferSize: 1024, format: fmt) { [weak self] buf, _ in
//            guard let self else { return }
//
//            if let data = buf.floatChannelData {
//                let n = Int(buf.frameLength)
//                var sum: Float = 0
//                for i in 0..<n { sum += data[0][i] * data[0][i] }
//                let rms = sqrt(sum / Float(n))
//                if rms > 0.001 {
//                    print(String(format: "TAP RMS: %.5f", rms))
//                }
//            }
//
//            let callNum = self.tapCallCount + 1
//            DispatchQueue.main.async { self.tapCallCount = callNum }
//            self.processTapBuffer(buffer: buf, tapCallNumber: callNum)
//        }
//        tapNode      = node
//        tapInstalled = true
//        print("=== Tap installed.")
//    }
//
//    private func removeTap() {
//        guard tapInstalled, let node = tapNode else { return }
//        node.removeTap(onBus: 0)
//        tapNode      = nil
//        tapInstalled = false
//        print("=== Tap removed.")
//    }
//
//    // MARK: - DSP
//
//    private func processTapBuffer(buffer: AVAudioPCMBuffer, tapCallNumber: Int) {
//        guard let channelData = buffer.floatChannelData else {
//            print("=== processTapBuffer #\(tapCallNumber): no channelData")
//            return
//        }
//        let frameLength = Int(buffer.frameLength)
//        guard frameLength > 0 else {
//            print("=== processTapBuffer #\(tapCallNumber): frameLength=0")
//            return
//        }
//
//        let ch   = Int(buffer.format.channelCount)
//        var mono = [Double](repeating: 0, count: frameLength)
//        if ch >= 2 {
//            let c0 = channelData[0], c1 = channelData[1]
//            for i in 0..<frameLength { mono[i] = Double(c0[i] + c1[i]) * 0.5 }
//        } else {
//            let c0 = channelData[0]
//            for i in 0..<frameLength { mono[i] = Double(c0[i]) }
//        }
//
//        if tapCallNumber <= 5 {
//            let monoRMS = sqrt(mono.map{$0*$0}.reduce(0,+) / Double(mono.count))
//            print(String(format: "=== processTapBuffer #%d  frames=%d  ch=%d  monoRMS=%.6f",
//                         tapCallNumber, frameLength, ch, monoRMS))
//        }
//
//        let raw: [AnyHashable: Any] = mono.withUnsafeBufferPointer { ptr in
//            self.engineWrapper.processBlock(ptr.baseAddress!, length: frameLength)
//        }
//
//        if tapCallNumber <= 10 {
//            print("=== processBlock returned \(raw.count) keys: \(raw.keys.map{"\($0)"})")
//            if let conf = raw["confidence"] as? NSNumber {
//                print(String(format: "    confidence=%.4f", conf.doubleValue))
//            }
//            if let pos = raw["position"] as? NSNumber {
//                print("    position=\(pos.intValue)")
//            }
//        }
//
//        if !raw.isEmpty {
//            DispatchQueue.main.async {
//                self.processCallCount += 1
//                self.lastResultKeys = raw.keys.compactMap { $0 as? String }
//            }
//        }
//
//        DispatchQueue.main.async { self.unpackStatus(raw) }
//    }
//
//    // MARK: - Unpack
//
//    private func unpackStatus(_ s: [AnyHashable: Any]) {
//        if let n = s["ref_time"]        as? NSNumber { refTime       = n.doubleValue }
//        if let n = s["progress"]        as? NSNumber { progress      = n.doubleValue }
//        if let n = s["confidence"]      as? NSNumber { confidence    = n.doubleValue }
//        if let n = s["overall_gain_db"] as? NSNumber { overallGainDB = n.doubleValue }
//        unpackDict(s["stemGains"], into: &stemGains)
//        unpackDict(s["stemLive"],  into: &stemLiveDB)
//        unpackDict(s["stemRef"],   into: &stemRefDB)
//    }
//
//    private func unpackDict(_ raw: Any?, into target: inout [String: Double]) {
//        if let d = raw as? [String: NSNumber] {
//            for (k, v) in d { target[k] = v.doubleValue }
//        } else if let d = raw as? [String: Any] {
//            for (k, v) in d { if let n = v as? NSNumber { target[k] = n.doubleValue } }
//        }
//    }
//}
