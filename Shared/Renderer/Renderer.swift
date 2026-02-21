import MetalKit
import MetalFX
import simd

/// Core renderer — coordinates Metal 4 rendering, neural weight buffers, and the game loop.
@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let gameState: GameState
    let neuralWeights: NeuralWeights

    // Pipeline states
    private var renderPipelineState: MTLRenderPipelineState!
    private var terrainComputePipeline: MTLComputePipelineState!
    private var colorComputePipeline: MTLComputePipelineState!

    // Mesh
    private var vertexBuffer: MTLBuffer!
    private var indexBuffer: MTLBuffer!
    private var indexCount: Int = 0

    // Depth
    private var depthStencilState: MTLDepthStencilState!

    // Uniforms
    private var uniformBuffer: MTLBuffer!
    private var playerStateBuffer: MTLBuffer!

    // Compute shader parameters
    private var deltaTimeBuffer: MTLBuffer!
    private var decayRateBuffer: MTLBuffer!

    // MetalFX upscaling
    private let upscaler: MetalFXUpscaler
    private var blitPipelineState: MTLRenderPipelineState!

    let decayRate: Float = 0.02
    private var frameCount = 0

    init?(metalView: MTKView) {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            return nil
        }

        self.device = device
        self.commandQueue = commandQueue
        self.gameState = GameState()
        self.neuralWeights = NeuralWeights(device: device)
        self.upscaler = MetalFXUpscaler(device: device, upscaleFactor: 1.5)

        super.init()

        metalView.device = device
        metalView.colorPixelFormat = .bgra8Unorm_srgb
        metalView.depthStencilPixelFormat = .depth32Float
        metalView.clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1.0)
        metalView.preferredFramesPerSecond = 60
        metalView.delegate = self
        metalView.framebufferOnly = false // Needed for MetalFX blit

        buildPipelines(metalView: metalView)
        buildBlitPipeline(metalView: metalView)
        buildMesh()
        buildBuffers()
        buildDepthState()
    }

    // MARK: - Pipeline Setup

    private func buildPipelines(metalView: MTKView) {
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Failed to create default Metal library")
        }

        // Render pipeline
        let vertexFunction = library.makeFunction(name: "neuralTerrainVertex")
        let fragmentFunction = library.makeFunction(name: "neuralColorFragment")

        let pipelineDescriptor = MTLRenderPipelineDescriptor()
        pipelineDescriptor.label = "Neural Terrain Pipeline"
        pipelineDescriptor.vertexFunction = vertexFunction
        pipelineDescriptor.fragmentFunction = fragmentFunction
        pipelineDescriptor.colorAttachments[0].pixelFormat = metalView.colorPixelFormat
        pipelineDescriptor.depthAttachmentPixelFormat = metalView.depthStencilPixelFormat

        // Vertex descriptor matching GridVertex struct (sizeof = 32 in Metal)
        let vertexDescriptor = MTLVertexDescriptor()
        // Position: float3 at offset 0
        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0
        // Texcoord: float2 at offset 16
        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = 16
        vertexDescriptor.attributes[1].bufferIndex = 0
        // Layout: stride must match Metal's sizeof(GridVertex) = 32
        vertexDescriptor.layouts[0].stride = 32
        vertexDescriptor.layouts[0].stepFunction = .perVertex

        pipelineDescriptor.vertexDescriptor = vertexDescriptor

        do {
            renderPipelineState = try device.makeRenderPipelineState(descriptor: pipelineDescriptor)
        } catch {
            fatalError("Failed to create render pipeline state: \(error)")
        }

        // Compute pipelines for weight updates
        if let terrainKernel = library.makeFunction(name: "updateTerrainWeights") {
            terrainComputePipeline = try? device.makeComputePipelineState(function: terrainKernel)
        }
        if let colorKernel = library.makeFunction(name: "updateColorWeights") {
            colorComputePipeline = try? device.makeComputePipelineState(function: colorKernel)
        }
    }

    private func buildBlitPipeline(metalView: MTKView) {
        guard let library = device.makeDefaultLibrary(),
              let vertexFunc = library.makeFunction(name: "blitVertex"),
              let fragFunc = library.makeFunction(name: "blitFragment") else { return }

        let desc = MTLRenderPipelineDescriptor()
        desc.label = "Blit Pipeline"
        desc.vertexFunction = vertexFunc
        desc.fragmentFunction = fragFunc
        desc.colorAttachments[0].pixelFormat = metalView.colorPixelFormat

        blitPipelineState = try? device.makeRenderPipelineState(descriptor: desc)
    }

    // MARK: - Mesh Generation

    private func buildMesh() {
        let gridSize = gameState.gridSize
        let spacing = gameState.gridSpacing
        let halfExtent = Float(gridSize) * spacing * 0.5

        // Metal's GridVertex: float3(16) + float2(8) = 24 data bytes,
        // but struct alignment is 16 (from float3), so sizeof = 32.
        let vertexCount = (gridSize + 1) * (gridSize + 1)
        let metalVertexStride = 32  // Must match sizeof(GridVertex) in Metal
        var vertexData = [UInt8](repeating: 0, count: vertexCount * metalVertexStride)

        vertexData.withUnsafeMutableBytes { rawBuffer in
            var offset = 0
            for z in 0...gridSize {
                for x in 0...gridSize {
                    let px = Float(x) * spacing - halfExtent
                    let pz = Float(z) * spacing - halfExtent
                    let position = SIMD3<Float>(px, 0, pz)
                    let texcoord = SIMD2<Float>(Float(x) / Float(gridSize),
                                                Float(z) / Float(gridSize))

                    // float3 position at offset+0 (16 bytes)
                    rawBuffer.storeBytes(of: position, toByteOffset: offset, as: SIMD3<Float>.self)
                    // float2 texcoord at offset+16 (8 bytes)
                    rawBuffer.storeBytes(of: texcoord, toByteOffset: offset + 16, as: SIMD2<Float>.self)
                    // 8 bytes of padding at offset+24 (implicit zeros)
                    offset += metalVertexStride
                }
            }
        }

        vertexBuffer = device.makeBuffer(bytes: vertexData,
                                          length: vertexData.count,
                                          options: .storageModeShared)
        vertexBuffer.label = "Terrain Vertex Buffer"

        // Indices: two triangles per grid cell
        let cellCount = gridSize * gridSize
        indexCount = cellCount * 6
        var indices = [UInt32](repeating: 0, count: indexCount)

        var idx = 0
        let stride = UInt32(gridSize + 1)
        for z in 0..<UInt32(gridSize) {
            for x in 0..<UInt32(gridSize) {
                let topLeft = z * stride + x
                let topRight = topLeft + 1
                let bottomLeft = (z + 1) * stride + x
                let bottomRight = bottomLeft + 1

                indices[idx] = topLeft;     idx += 1
                indices[idx] = bottomLeft;  idx += 1
                indices[idx] = topRight;    idx += 1
                indices[idx] = topRight;    idx += 1
                indices[idx] = bottomLeft;  idx += 1
                indices[idx] = bottomRight; idx += 1
            }
        }

        indexBuffer = device.makeBuffer(bytes: indices,
                                         length: indices.count * MemoryLayout<UInt32>.size,
                                         options: .storageModeShared)
        indexBuffer.label = "Terrain Index Buffer"
    }

    // MARK: - Buffers

    private func buildBuffers() {
        uniformBuffer = device.makeBuffer(length: MemoryLayout<Uniforms>.stride,
                                           options: .storageModeShared)
        uniformBuffer.label = "Uniforms"

        playerStateBuffer = device.makeBuffer(length: MemoryLayout<PlayerStateGPU>.stride,
                                               options: .storageModeShared)
        playerStateBuffer.label = "Player State"

        deltaTimeBuffer = device.makeBuffer(length: MemoryLayout<Float>.size,
                                             options: .storageModeShared)
        deltaTimeBuffer.label = "Delta Time"

        decayRateBuffer = device.makeBuffer(length: MemoryLayout<Float>.size,
                                             options: .storageModeShared)
        decayRateBuffer.label = "Decay Rate"

        decayRateBuffer.contents().storeBytes(of: decayRate, as: Float.self)
    }

    private func buildDepthState() {
        let descriptor = MTLDepthStencilDescriptor()
        descriptor.depthCompareFunction = .less
        descriptor.isDepthWriteEnabled = true
        depthStencilState = device.makeDepthStencilState(descriptor: descriptor)
    }

    // MARK: - Frame Update

    private func updateUniforms(drawableSize: CGSize) {
        let aspectRatio = Float(drawableSize.width / drawableSize.height)
        let projection = gameState.projectionMatrix(aspectRatio: aspectRatio)
        let view = gameState.viewMatrix
        let modelView = view // Model is identity
        let mvp = projection * modelView

        // Extract upper-left 3x3 for normal matrix
        let col0 = SIMD3<Float>(modelView[0][0], modelView[0][1], modelView[0][2])
        let col1 = SIMD3<Float>(modelView[1][0], modelView[1][1], modelView[1][2])
        let col2 = SIMD3<Float>(modelView[2][0], modelView[2][1], modelView[2][2])
        let normalMatrix = simd_float3x3(col0, col1, col2)

        var uniforms = Uniforms(
            modelViewProjection: mvp,
            modelView: modelView,
            normalMatrix: normalMatrix,
            cameraPosition: gameState.cameraPosition,
            time: gameState.totalTime,
            gridSize: Float(gameState.gridSize),
            gridSpacing: gameState.gridSpacing,
            _padding: 0
        )

        uniformBuffer.contents().copyMemory(from: &uniforms,
                                             byteCount: MemoryLayout<Uniforms>.stride)
    }

    private func updatePlayerState() {
        var ps = PlayerStateGPU(
            position: gameState.cameraPosition,
            influenceRadius: gameState.influenceRadius,
            interactionStrength: gameState.interactionStrength,
            isInteracting: gameState.isInteracting ? 1 : 0,
            _padding: (0, 0)
        )
        playerStateBuffer.contents().copyMemory(from: &ps,
                                                 byteCount: MemoryLayout<PlayerStateGPU>.stride)

        deltaTimeBuffer.contents().storeBytes(of: gameState.deltaTime, as: Float.self)
    }

    // MARK: - Weight Update (Compute Pass)

    private func encodeWeightUpdate(commandBuffer: MTLCommandBuffer) {
        guard let terrainPipeline = terrainComputePipeline,
              let colorPipeline = colorComputePipeline else { return }

        guard let computeEncoder = commandBuffer.makeComputeCommandEncoder() else { return }
        computeEncoder.label = "Weight Update"

        // Terrain weights
        computeEncoder.setComputePipelineState(terrainPipeline)
        computeEncoder.setBuffer(neuralWeights.terrainWeights, offset: 0, index: 0)
        computeEncoder.setBuffer(playerStateBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(deltaTimeBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(decayRateBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(neuralWeights.terrainInitialWeights, offset: 0, index: 4)

        let terrainThreads = MTLSize(width: NeuralWeights.terrainCount, height: 1, depth: 1)
        let terrainThreadgroup = MTLSize(width: min(256, NeuralWeights.terrainCount), height: 1, depth: 1)
        computeEncoder.dispatchThreads(terrainThreads, threadsPerThreadgroup: terrainThreadgroup)

        // Color weights
        computeEncoder.setComputePipelineState(colorPipeline)
        computeEncoder.setBuffer(neuralWeights.colorWeights, offset: 0, index: 0)
        computeEncoder.setBuffer(playerStateBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(deltaTimeBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(decayRateBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(neuralWeights.colorInitialWeights, offset: 0, index: 4)

        let colorThreads = MTLSize(width: NeuralWeights.colorCount, height: 1, depth: 1)
        let colorThreadgroup = MTLSize(width: min(256, NeuralWeights.colorCount), height: 1, depth: 1)
        computeEncoder.dispatchThreads(colorThreads, threadsPerThreadgroup: colorThreadgroup)

        computeEncoder.endEncoding()
    }

    // MARK: - Terrain Render Encoding (shared between upscaled and direct paths)

    private func encodeTerrainRender(encoder: MTLRenderCommandEncoder) {
        encoder.label = "Neural Terrain Render"
        encoder.setRenderPipelineState(renderPipelineState)
        encoder.setDepthStencilState(depthStencilState)
        encoder.setFrontFacing(.counterClockwise)
        encoder.setCullMode(.none)

        // Vertex shader buffers
        encoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        encoder.setVertexBuffer(uniformBuffer, offset: 0, index: 1)
        encoder.setVertexBuffer(neuralWeights.terrainWeights, offset: 0, index: 2)
        encoder.setVertexBuffer(playerStateBuffer, offset: 0, index: 4)

        // Fragment shader buffers
        encoder.setFragmentBuffer(neuralWeights.colorWeights, offset: 0, index: 3)
        encoder.setFragmentBuffer(playerStateBuffer, offset: 0, index: 4)

        encoder.drawIndexedPrimitives(
            type: .triangle,
            indexCount: indexCount,
            indexType: .uint32,
            indexBuffer: indexBuffer,
            indexBufferOffset: 0
        )

        encoder.endEncoding()
    }

    // MARK: - MTKViewDelegate

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        upscaler.resize(outputSize: size, colorPixelFormat: view.colorPixelFormat)
    }

    func draw(in view: MTKView) {
        gameState.update()
        updatePlayerState()

        guard let commandBuffer = commandQueue.makeCommandBuffer(),
              let drawable = view.currentDrawable else { return }

        frameCount += 1
        if frameCount == 1 {
            print("[Renderer] First frame. drawableSize=\(view.drawableSize), indexCount=\(indexCount)")
            print("[Renderer] MetalFX ready=\(upscaler.isReady), blitPipeline=\(blitPipelineState != nil)")
            print("[Renderer] Uniforms stride=\(MemoryLayout<Uniforms>.stride), size=\(MemoryLayout<Uniforms>.size)")
            print("[Renderer] PlayerState stride=\(MemoryLayout<PlayerStateGPU>.stride), size=\(MemoryLayout<PlayerStateGPU>.size)")
        }

        // Compute pass: update weights
        encodeWeightUpdate(commandBuffer: commandBuffer)

        // Ensure MetalFX textures are created (resize may not fire before first draw)
        if !upscaler.isReady {
            upscaler.resize(outputSize: view.drawableSize, colorPixelFormat: view.colorPixelFormat)
        }

        // --- MetalFX Path: render low-res → upscale → blit to drawable ---
        if upscaler.isReady, let blitPipeline = blitPipelineState {
            if frameCount == 1 { print("[Renderer] Using MetalFX path") }
            let renderSize = CGSize(width: upscaler.renderWidth, height: upscaler.renderHeight)
            updateUniforms(drawableSize: renderSize)

            // 1. Render terrain to offscreen low-res target
            guard let offscreenRPD = upscaler.makeRenderPassDescriptor(),
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: offscreenRPD) else { return }
            encodeTerrainRender(encoder: renderEncoder)

            // 2. MetalFX spatial upscale
            upscaler.encode(commandBuffer: commandBuffer)

            // 3. Blit upscaled output to drawable
            let blitRPD = MTLRenderPassDescriptor()
            blitRPD.colorAttachments[0].texture = drawable.texture
            blitRPD.colorAttachments[0].loadAction = .dontCare
            blitRPD.colorAttachments[0].storeAction = .store

            guard let blitEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: blitRPD) else { return }
            blitEncoder.label = "Blit Upscaled"
            blitEncoder.setRenderPipelineState(blitPipeline)
            blitEncoder.setFragmentTexture(upscaler.outputTexture, index: 0)
            blitEncoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            blitEncoder.endEncoding()

        } else {
            // --- Fallback: render directly to drawable ---
            if frameCount == 1 { print("[Renderer] Using FALLBACK path (direct to drawable)") }
            updateUniforms(drawableSize: view.drawableSize)
            guard let renderPassDescriptor = view.currentRenderPassDescriptor,
                  let renderEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) else { return }
            encodeTerrainRender(encoder: renderEncoder)
        }

        commandBuffer.present(drawable)
        commandBuffer.commit()
    }

    func resetWorld() {
        neuralWeights.reset()
    }
}

// MARK: - GPU-compatible struct (matches PlayerState in ShaderTypes.h)

// NOTE: Swift's SIMD3<Float> has stride 16 inside structs, matching Metal's float3.
// No extra padding needed between SIMD3 and subsequent fields.

struct PlayerStateGPU {
    var position: SIMD3<Float>
    var influenceRadius: Float
    var interactionStrength: Float
    var isInteracting: Int32
    var _padding: (Float, Float)
}

// MARK: - Uniforms struct (matches ShaderTypes.h)

struct Uniforms {
    var modelViewProjection: simd_float4x4
    var modelView: simd_float4x4
    var normalMatrix: simd_float3x3
    var cameraPosition: SIMD3<Float>
    var time: Float
    var gridSize: Float
    var gridSpacing: Float
    var _padding: Float
}
