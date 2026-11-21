// surface_run: IOSurface handoff test for weightless Core ML ANE programs.
//
// Design follows the technique in ~/zCode_26_11/drawthings-ane-research/
// DRAWTHINGS-ANE-FINDINGS.md, itself describing liuliu/ccv (BSD-3-Clause,
// copyright (c) Li Liu). This file is written from that description; it
// copies no ccv source. Keep this notice with derived work.
//
// One process holds every shape program plus ONE set of shared IOSurfaces
// (weights, activations, output), cycled across L synthetic layers.
// Per call: copy the layer's int8 weights into the shared weight surface
// (memcpy mode, or Metal blit with --blit), then predict synchronously with
// outputBackings onto the shared output surface. Reports per-call memcpy ms,
// predict ms, resident size; the driver samples wired memory from outside.
//
// Usage: surface_run <M> [--layers L] [--iters T] [--blit] model N K [...]

import Accelerate
import CoreML
import Foundation
import Metal
import QuartzCore

let argv = CommandLine.arguments
guard argv.count >= 5 else {
    print("{\"error\":\"usage: surface_run M [--layers L] [--iters T] [--blit] model N K ...\"}")
    exit(2)
}
var pos = 2
let Mpad = Int(argv[1])!
var layers = 34, iters = 5, blitMode = false
while pos < argv.count, argv[pos].hasPrefix("--") {
    switch argv[pos] {
    case "--layers": pos += 1; layers = Int(argv[pos])!
    case "--iters": pos += 1; iters = Int(argv[pos])!
    case "--blit": blitMode = true
    case "--selfcheck": break
    default: print("{\"error\":\"bad flag \(argv[pos])\"}"); exit(2)
    }
    pos += 1
}
var sites: [(path: String, N: Int, K: Int)] = []
while pos + 2 <= argv.count, !argv[pos].hasPrefix("--") {
    sites.append((argv[pos], Int(argv[pos + 1])!, Int(argv[pos + 2])!))
    pos += 3
}
precondition(!sites.isEmpty, "no sites")

func elem(_ dt: MLMultiArrayDataType) -> Int { dt == .float16 ? 2 : 1 }

func makeSurface(_ width: Int, _ height: Int, _ half: Bool) -> CVPixelBuffer {
    var pb: CVPixelBuffer?
    let fmt = half ? kCVPixelFormatType_OneComponent16Half : kCVPixelFormatType_OneComponent8
    let attrs: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:],
        kCVPixelBufferMetalCompatibilityKey: true,
        kCVPixelBufferBytesPerRowAlignmentKey: 64,
    ]
    let e = CVPixelBufferCreate(kCFAllocatorDefault, width, height, fmt, attrs as CFDictionary, &pb)
    precondition(e == kCVReturnSuccess && pb != nil, "pixelbuffer \(width)x\(height) err \(e)")
    let b = pb!
    CVPixelBufferLockBaseAddress(b, [])
    let bpr = CVPixelBufferGetBytesPerRow(b)
    CVPixelBufferUnlockBaseAddress(b, [])
    precondition(bpr == width * (half ? 2 : 1), "rows not packed: \(bpr) != \(width * (half ? 2 : 1))")
    return b
}

func multi(_ pb: CVPixelBuffer, _ rows: Int, _ cols: Int, _ dt: MLMultiArrayDataType) -> MLMultiArray {
    // Model tensors are 4-D [1,1,rows,cols]; the pixel-buffer initializer needs that full shape.
    _ = dt
    return MLMultiArray(pixelBuffer: pb,
                        shape: [NSNumber(value: 1), NSNumber(value: 1), NSNumber(value: rows), NSNumber(value: cols)])
}

func loadModel(_ path: String) -> MLModel {
    let url = URL(fileURLWithPath: path)
    let cfg = MLModelConfiguration()
    cfg.computeUnits = .cpuAndNeuralEngine
    let compiled = try! MLModel.compileModel(at: url)
    return try! MLModel(contentsOf: compiled, configuration: cfg)
}

var device: MTLDevice?
var commandQueue: MTLCommandQueue?
if blitMode {
    device = MTLCreateSystemDefaultDevice()
    commandQueue = device!.makeCommandQueue()
}

struct Site { let model: MLModel; let N: Int; let K: Int; let wSurf: CVPixelBuffer
              let xSurf: CVPixelBuffer; let ySurf: CVPixelBuffer; let wArr: MLMultiArray
              let xArr: MLMultiArray; let yArr: MLMultiArray }

var siteList: [Site] = []
for s in sites {
    let wSurf = makeSurface(s.K, s.N, false)
    let xSurf = makeSurface(Mpad, s.K, false)
    let ySurf = makeSurface(Mpad, s.N, true)
    siteList.append(Site(model: loadModel(s.path), N: s.N, K: s.K, wSurf: wSurf, xSurf: xSurf,
                         ySurf: ySurf,
                         wArr: multi(wSurf, s.N, s.K, .int8),
                         xArr: multi(xSurf, s.K, Mpad, .int8),
                         yArr: multi(ySurf, s.N, Mpad, .float16)))
}

// Layer weight blobs: deterministic int8 per (layer, site).
var blobs: [[UnsafeMutableRawPointer]] = []
for layer in 0..<layers {
    var per: [UnsafeMutableRawPointer] = []
    for s in siteList {
        let n = s.N * s.K
        let p = UnsafeMutableRawPointer.allocate(byteCount: n, alignment: 64)
        let bytes = p.bindMemory(to: UInt8.self, capacity: n)
        var state = UInt64(layer &* 1_000_003 &+ s.N &* 7 &+ 11)  // xorshift, deterministic
        for i in stride(from: 0, to: n, by: 8) {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            let v: Int8 = Int8(truncatingIfNeeded: state >> 27)
            bytes[i] = UInt8(bitPattern: v & 0x7F)
            if i + 1 < n { bytes[i + 1] = UInt8(bitPattern: v) }
        }
        per.append(p)
    }
    blobs.append(per)
}
// Metal views cached per site/blob for blit mode.
var dstBufs: [MTLBuffer] = []
var srcBufs: [[MTLBuffer]] = []
if blitMode {
    for s in siteList {
        CVPixelBufferLockBaseAddress(s.wSurf, [])
        let n = s.N * s.K
        dstBufs.append(device!.makeBuffer(bytesNoCopy: CVPixelBufferGetBaseAddress(s.wSurf)!,
                                          length: n, options: [], deallocator: nil)!)
        CVPixelBufferUnlockBaseAddress(s.wSurf, [])
    }
    for per in blobs {
        var rowBufs: [MTLBuffer] = []
        for (si, p) in per.enumerated() {
            rowBufs.append(device!.makeBuffer(bytesNoCopy: p, length: siteList[si].N * siteList[si].K,
                                              options: [], deallocator: nil)!)
        }
        srcBufs.append(rowBufs)
    }
}

// Fixed activation int8 per site.
for s in siteList {
    CVPixelBufferLockBaseAddress(s.xSurf, [])
    let base = CVPixelBufferGetBaseAddress(s.xSurf)!.bindMemory(to: UInt8.self, capacity: s.K * Mpad)
    for i in 0..<(s.K * Mpad) { base[i] = UInt8(truncatingIfNeeded: (i * 37) & 0x3F &- 31) }
    CVPixelBufferUnlockBaseAddress(s.xSurf, [])
}

func blitWeight(_ si: Int, _ layer: Int) -> Double {
    let q = commandQueue!
    let n = siteList[si].N * siteList[si].K
    let cb = q.makeCommandBuffer()!
    let enc = cb.makeBlitCommandEncoder()!
    enc.copy(from: srcBufs[layer][si], sourceOffset: 0, to: dstBufs[si], destinationOffset: 0, size: n)
    enc.endEncoding()
    let t0 = DispatchTime.now().uptimeNanoseconds
    cb.commit()
    cb.waitUntilCompleted()
    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}

func copyWeight(_ site: Site, _ src: UnsafeMutableRawPointer) -> Double {
    let n = site.N * site.K
    CVPixelBufferLockBaseAddress(site.wSurf, [])
    let t0 = DispatchTime.now().uptimeNanoseconds
    memcpy(CVPixelBufferGetBaseAddress(site.wSurf)!, src, n)
    let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    CVPixelBufferUnlockBaseAddress(site.wSurf, [])
    return dt
}

func predict(_ site: Site) throws -> Double {
    let feats = try MLDictionaryFeatureProvider(dictionary: [
        "w": MLFeatureValue(multiArray: site.wArr),
        "x": MLFeatureValue(multiArray: site.xArr),
    ])
    let opts = MLPredictionOptions()
    opts.outputBackings = ["y": site.yArr]
    let t0 = DispatchTime.now().uptimeNanoseconds
    _ = try site.model.prediction(from: feats, options: opts)
    return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
}

// Warm the first layer of every site once (cold ANE plans included).
for (si, s) in siteList.enumerated() {
    _ = blitMode ? blitWeight(si, 0) : copyWeight(s, blobs[0][si])
    let t = try predict(s)
    print("{\"event\":\"cold\",\"site\":\(si),\"predict_ms\":\(String(format: "%.2f", t))}")
}

if CommandLine.arguments.contains("--selfcheck") {
    // Known-answer check on the surface path: w codes 1, x codes 2 -> y = K*2/K = 2.0.
    let st = siteList[0]
    func fill(_ pb: CVPixelBuffer, _ v: UInt8) {
        CVPixelBufferLockBaseAddress(pb, [])
        let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
        let n = CVPixelBufferGetBytesPerRow(pb) * CVPixelBufferGetHeight(pb)
        for i in 0..<n { base[i] = v }
        CVPixelBufferUnlockBaseAddress(pb, [])
    }
    fill(st.wSurf, 1)
    fill(st.xSurf, 2)
    _ = try! predict(st)
    CVPixelBufferLockBaseAddress(st.ySurf, [])
    let yb = CVPixelBufferGetBaseAddress(st.ySurf)!.assumingMemoryBound(to: Float16.self)
    let v0 = Double(yb[0]), v1 = Double(yb[1])
    CVPixelBufferUnlockBaseAddress(st.ySurf, [])
    print("{\"event\":\"selfcheck\",\"y0\":\(v0),\"y1\":\(v1),\"pass\":\(abs(v0 - 2.0) < 0.01 && abs(v1 - 2.0) < 0.01)}")
}

var copyTimes: [Double] = [], predictTimes: [Double] = []
for it in 0..<iters {
    for layer in 0..<layers {
        for (si, s) in siteList.enumerated() {
            let c = blitMode ? blitWeight(si, layer) : copyWeight(s, blobs[layer][si])
            let p = try predict(s)
            copyTimes.append(c)
            predictTimes.append(p)
        }
    }
    print("{\"event\":\"mark\",\"iter\":\(it)}")  // external sampler boundary
}

copyTimes.sort(); predictTimes.sort()
let count = predictTimes.count
struct Out: Encodable { let mode: String; let layers: Int; let sites: Int; let calls: Int
    let copy_med_ms: Double; let copy_p90_ms: Double
    let pred_med_ms: Double; let pred_p90_ms: Double }
let o = Out(mode: blitMode ? "blit" : "memcpy", layers: layers, sites: sites.count, calls: count,
            copy_med_ms: copyTimes[count / 2], copy_p90_ms: copyTimes[Int(Double(count) * 0.9)],
            pred_med_ms: predictTimes[count / 2], pred_p90_ms: predictTimes[Int(Double(count) * 0.9)])
let d = try JSONEncoder().encode([o])
FileHandle.standardOutput.write(d)
