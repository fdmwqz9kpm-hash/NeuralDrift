import Foundation
import AVFoundation

/// Procedural ambient audio that reacts to neural network state.
/// Uses AVAudioEngine with tone generators — no audio files needed.
/// Maps weight variance → drone pitch, mutation → transient sounds,
/// resonance capture → chime.
final class AudioEngine {
    private let engine = AVAudioEngine()
    private let mixer = AVAudioMixerNode()

    // Tone generators
    private var droneNode: AVAudioSourceNode?
    private var mutationNode: AVAudioSourceNode?
    private var chimeNode: AVAudioSourceNode?

    // State
    private var droneFrequency: Float = 80.0
    private var droneTargetFreq: Float = 80.0
    private var droneVolume: Float = 0.12
    private var mutationPhase: Float = 0
    private var mutationEnvelope: Float = 0
    private var chimePhase: Float = 0
    private var chimeEnvelope: Float = 0
    private var chimePitch: Float = 440.0

    // Harmonic content
    private var dronePhase1: Float = 0
    private var dronePhase2: Float = 0
    private var dronePhase3: Float = 0

    private let sampleRate: Float = 44100.0
    private var isRunning = false

    init() {
        setupAudio()
    }

    deinit {
        stop()
    }

    // MARK: - Public API

    func start() {
        guard !isRunning else { return }
        do {
            try engine.start()
            isRunning = true
        } catch {
            print("[Audio] Failed to start: \(error)")
        }
    }

    func stop() {
        guard isRunning else { return }
        engine.stop()
        isRunning = false
    }

    /// Update audio parameters from neural state. Call every frame.
    func update(weightVariance: Float, isInteracting: Bool, deltaTime: Float) {
        // Map weight variance to drone frequency (higher variance = higher pitch)
        droneTargetFreq = 60.0 + weightVariance * 120.0
        droneTargetFreq = min(max(droneTargetFreq, 40.0), 200.0)

        // Smooth frequency changes
        droneFrequency += (droneTargetFreq - droneFrequency) * min(1.0, 2.0 * deltaTime)

        // Mutation sound: trigger on interaction
        if isInteracting {
            mutationEnvelope = min(mutationEnvelope + deltaTime * 8.0, 1.0)
        } else {
            mutationEnvelope *= max(0, 1.0 - deltaTime * 3.0)
        }
    }

    /// Trigger a resonance capture chime
    func playChime(pitch: Float = 660.0) {
        chimePitch = pitch
        chimeEnvelope = 1.0
    }

    // MARK: - Setup

    private func setupAudio() {
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate),
                                    channels: 1)!

        // Ambient drone — deep harmonic hum
        let droneSource = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let buf = ablPointer[0]
            let frames = Int(frameCount)
            guard let data = buf.mData?.assumingMemoryBound(to: Float.self) else { return noErr }

            let freq = self.droneFrequency
            let vol = self.droneVolume
            let dt = 1.0 / self.sampleRate

            for i in 0..<frames {
                // Rich drone: fundamental + octave + fifth
                self.dronePhase1 += freq * dt
                self.dronePhase2 += freq * 2.0 * dt
                self.dronePhase3 += freq * 1.5 * dt

                if self.dronePhase1 > 1.0 { self.dronePhase1 -= 1.0 }
                if self.dronePhase2 > 1.0 { self.dronePhase2 -= 1.0 }
                if self.dronePhase3 > 1.0 { self.dronePhase3 -= 1.0 }

                let s1 = sin(self.dronePhase1 * .pi * 2.0) * 0.5
                let s2 = sin(self.dronePhase2 * .pi * 2.0) * 0.2
                let s3 = sin(self.dronePhase3 * .pi * 2.0) * 0.15

                data[i] = (s1 + s2 + s3) * vol
            }
            return noErr
        }

        // Mutation sound — gritty noise burst
        let mutSource = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let buf = ablPointer[0]
            let frames = Int(frameCount)
            guard let data = buf.mData?.assumingMemoryBound(to: Float.self) else { return noErr }

            let env = self.mutationEnvelope
            let dt = 1.0 / self.sampleRate

            for i in 0..<frames {
                self.mutationPhase += 220.0 * dt
                if self.mutationPhase > 1.0 { self.mutationPhase -= 1.0 }

                // Distorted sine + noise for gritty mutation feel
                let sine = sin(self.mutationPhase * .pi * 2.0)
                let noise = Float.random(in: -1...1) * 0.3
                let distorted = tanh(sine * 3.0) * 0.4 + noise

                data[i] = distorted * env * 0.15
            }
            return noErr
        }

        // Chime — bell-like tone for resonance capture
        let chimeSource = AVAudioSourceNode { [weak self] _, _, frameCount, audioBufferList -> OSStatus in
            guard let self = self else { return noErr }
            let ablPointer = UnsafeMutableAudioBufferListPointer(audioBufferList)
            let buf = ablPointer[0]
            let frames = Int(frameCount)
            guard let data = buf.mData?.assumingMemoryBound(to: Float.self) else { return noErr }

            let env = self.chimeEnvelope
            let freq = self.chimePitch
            let dt = 1.0 / self.sampleRate

            for i in 0..<frames {
                self.chimePhase += freq * dt
                if self.chimePhase > 1.0 { self.chimePhase -= 1.0 }

                // Bell: fundamental + minor third + fifth (inharmonic partials)
                let f1 = sin(self.chimePhase * .pi * 2.0)
                let f2 = sin(self.chimePhase * .pi * 2.0 * 2.4) * 0.3
                let f3 = sin(self.chimePhase * .pi * 2.0 * 3.0) * 0.15

                data[i] = (f1 + f2 + f3) * env * 0.25

                // Exponential decay
                self.chimeEnvelope *= (1.0 - dt * 2.5)
                if self.chimeEnvelope < 0.001 { self.chimeEnvelope = 0 }
            }
            return noErr
        }

        droneNode = droneSource
        mutationNode = mutSource
        chimeNode = chimeSource

        engine.attach(droneSource)
        engine.attach(mutSource)
        engine.attach(chimeSource)
        engine.attach(mixer)

        engine.connect(droneSource, to: mixer, format: format)
        engine.connect(mutSource, to: mixer, format: format)
        engine.connect(chimeSource, to: mixer, format: format)
        engine.connect(mixer, to: engine.mainMixerNode, format: format)

        mixer.outputVolume = 0.6
    }
}
