import Foundation
import AVFoundation
import Combine

/// Manages the iOS audio engine for real-time microphone input capture.
///
/// Sets up an `AVAudioEngine` tap on the microphone input node, converts
/// incoming Float32 samples to Float64, and provides a hook for feeding
/// audio data into the C++ DSP engine (ASEWrapper).
class AudioController: ObservableObject {
    private var engine = AVAudioEngine()
    private var isRunning = false
    
    // Placeholder for the Objective-C++ wrapper around the C++ analysis engine.
    // Uncomment and initialize when the ASEWrapper bridge is integrated.
    // private var aseEngine = ASEWrapper()
    
    /// Start the audio engine and install a real-time tap on the microphone input.
    func start() {
        let inputNode = engine.inputNode
        let format = inputNode.inputFormat(forBus: 0)
        
        // Install a tap to receive audio buffers from the microphone at 1024-sample intervals
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
    
    /// Stop the audio engine and remove the microphone tap.
    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        isRunning = false
    }
    
    /// Process a single audio buffer from the microphone tap.
    /// Converts Float32 PCM samples to Float64 for the C++ DSP engine.
    private func processBuffer(buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameLength = Int(buffer.frameLength)
        
        // The C++ DSP engine expects double-precision samples
        var doubleArray = [Double](repeating: 0.0, count: frameLength)
        for i in 0..<frameLength {
            doubleArray[i] = Double(channelData[i])
        }
        
        // Feed into the C++ engine wrapper:
        // aseEngine.processBlock(doubleArray)
    }
}