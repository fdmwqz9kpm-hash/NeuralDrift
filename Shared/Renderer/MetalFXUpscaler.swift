import MetalKit
import MetalFX

/// Manages MetalFX spatial upscaling — renders at lower resolution and upscales
/// to the display resolution for better performance with minimal visual loss.
@MainActor
final class MetalFXUpscaler {
    private let device: MTLDevice
    private var spatialScaler: MTLFXSpatialScaler?

    // Offscreen render targets (lower resolution)
    private(set) var colorTexture: MTLTexture?
    private(set) var depthTexture: MTLTexture?
    private(set) var outputTexture: MTLTexture?

    // Resolution
    private(set) var renderWidth: Int = 0
    private(set) var renderHeight: Int = 0
    private(set) var outputWidth: Int = 0
    private(set) var outputHeight: Int = 0

    let upscaleFactor: Float

    /// - Parameter upscaleFactor: Ratio of output to render resolution (e.g., 1.5 means render at 2/3 size)
    init(device: MTLDevice, upscaleFactor: Float = 1.5) {
        self.device = device
        self.upscaleFactor = upscaleFactor
    }

    /// Rebuilds render targets and scaler when the drawable size changes.
    func resize(outputSize: CGSize, colorPixelFormat: MTLPixelFormat) {
        let outW = Int(outputSize.width)
        let outH = Int(outputSize.height)

        guard outW > 0 && outH > 0 else { return }
        guard outW != outputWidth || outH != outputHeight else { return }

        outputWidth = outW
        outputHeight = outH
        renderWidth = max(1, Int(Float(outW) / upscaleFactor))
        renderHeight = max(1, Int(Float(outH) / upscaleFactor))

        // Color render target
        let colorDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorPixelFormat,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        colorDesc.usage = [.renderTarget, .shaderRead]
        colorDesc.storageMode = .private
        colorTexture = device.makeTexture(descriptor: colorDesc)
        colorTexture?.label = "MetalFX Color Input"

        // Depth render target
        let depthDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float,
            width: renderWidth,
            height: renderHeight,
            mipmapped: false
        )
        depthDesc.usage = [.renderTarget, .shaderRead]
        depthDesc.storageMode = .private
        depthTexture = device.makeTexture(descriptor: depthDesc)
        depthTexture?.label = "MetalFX Depth Input"

        // Output texture (full resolution)
        let outputDesc = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: colorPixelFormat,
            width: outputWidth,
            height: outputHeight,
            mipmapped: false
        )
        outputDesc.usage = [.renderTarget, .shaderRead, .shaderWrite]
        outputDesc.storageMode = .private
        outputTexture = device.makeTexture(descriptor: outputDesc)
        outputTexture?.label = "MetalFX Output"

        // Create spatial scaler
        let scalerDesc = MTLFXSpatialScalerDescriptor()
        scalerDesc.inputWidth = renderWidth
        scalerDesc.inputHeight = renderHeight
        scalerDesc.outputWidth = outputWidth
        scalerDesc.outputHeight = outputHeight
        scalerDesc.colorTextureFormat = colorPixelFormat
        scalerDesc.outputTextureFormat = colorPixelFormat
        scalerDesc.colorProcessingMode = .perceptual

        spatialScaler = scalerDesc.makeSpatialScaler(device: device)
    }

    /// Returns a render pass descriptor targeting the offscreen low-res textures.
    func makeRenderPassDescriptor() -> MTLRenderPassDescriptor? {
        guard let color = colorTexture, let depth = depthTexture else { return nil }

        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = color
        rpd.colorAttachments[0].loadAction = .clear
        rpd.colorAttachments[0].storeAction = .store
        rpd.colorAttachments[0].clearColor = MTLClearColor(red: 0.02, green: 0.02, blue: 0.05, alpha: 1.0)
        rpd.depthAttachment.texture = depth
        rpd.depthAttachment.loadAction = .clear
        rpd.depthAttachment.storeAction = .store
        rpd.depthAttachment.clearDepth = 1.0

        return rpd
    }

    /// Encodes the MetalFX upscale pass into the command buffer.
    func encode(commandBuffer: MTLCommandBuffer) {
        guard let scaler = spatialScaler,
              let color = colorTexture,
              let output = outputTexture else { return }

        scaler.colorTexture = color
        scaler.outputTexture = output
        scaler.encode(commandBuffer: commandBuffer)
    }

    /// Whether MetalFX is available and configured.
    var isReady: Bool {
        spatialScaler != nil && colorTexture != nil
    }
}
