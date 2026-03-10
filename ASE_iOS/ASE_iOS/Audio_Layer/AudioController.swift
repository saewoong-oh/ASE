import Foundation
import AVFoundation
import Combine

class AudioController: ObservableObject {
    private var engine = AVAudioEngine()
    private var isRunning = false
    
    // This will interface with an Objective-C++ wrapper around your C++ engine
    // private var aseEngine = ASEWrapper()
    
    func start() {
        let inputNode = engine.inputNode
        let format = inputNode.inputFormat(forBus: 0)
        
        // Tap the microphone input
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] (buffer, time) in
            self?.processBuffer(buffer: buffer)
        }
        
        do {
            try engine.start()
            isRunning = true
            print("Audio engine started")
        } catch {
            print("Failed to start audio engine: \(error)")
        }
    }
    
    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
    }
    
    private func processBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)
        
        // Convert Float to Double for your C++ DSP engine
        var doubleArray = [Double](repeating: 0.0, count: frameLength)
        for i in 0..<frameLength {
            doubleArray[i] = Double(channelData[i])
        }
        
        // Feed into your C++ engine wrapper here:
        // aseEngine.processBlock(doubleArray)
    }
}
