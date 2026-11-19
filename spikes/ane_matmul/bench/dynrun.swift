// dynrun: predict/timing runner for dynamic-weights probe models.
// Usage: dynrun <model.mlpackage> <indir> [--burst N] [--wrap] [--cpu]
// Reads one raw binary file per model input named IN_<name>.bin, with the
// element type from the model description. --wrap tries zero-copy
// initWithDataPointer for int8 too (the question this probe answers);
// default path allocates an owned MLMultiArray and memcpy's.
import CoreML
import Foundation

let a = CommandLine.arguments
guard a.count >= 3 else { print("{\"error\":\"args\"}"); exit(1) }
let dir = a[1], indir = a[2]
var burst = 1, wrap = false, cpu = false
var i = 3
while i < a.count {
    switch a[i] {
    case "--burst": i += 1; burst = Int(a[i])!
    case "--wrap": wrap = true
    case "--cpu": cpu = true
    default: break
    }
    i += 1
}

func elemSize(_ d: MLMultiArrayDataType) -> Int {
    switch d {
    case .float16: return 2
    case .float32, .int32: return 4
    case .float64: return 8
    case .int8: return 1
    default: return 2
    }
}

func load(_ url: URL) throws -> MLModel {
    let cfg = MLModelConfiguration()
    cfg.computeUnits = cpu ? .cpuOnly : .cpuAndNeuralEngine
    let compiled = try MLModel.compileModel(at: url)
    return try MLModel(contentsOf: compiled, configuration: cfg)
}

let model = try load(URL(fileURLWithPath: dir))
var feats: [String: MLFeatureValue] = [:]
var kept: [UnsafeMutableRawPointer] = []  // wrap mode: buffers must outlive predict
for (name, desc) in model.modelDescription.inputDescriptionsByName {
    let cons = desc.multiArrayConstraint!
    let dt = cons.dataType
    let shape = cons.shape.map { $0.intValue }
    let n = shape.reduce(1, *)
    let data = try Data(contentsOf: URL(fileURLWithPath: "\(indir)/IN_\(name).bin"))
    precondition(data.count == n * elemSize(dt), "size mismatch \(name)")
    let arr: MLMultiArray
    var ptr: UnsafeMutableRawPointer? = nil
    if wrap {
        ptr = .allocate(byteCount: data.count, alignment: 4096)
        data.copyBytes(to: ptr!.assumingMemoryBound(to: UInt8.self), count: data.count)
        var strides: [NSNumber] = []
        var s = 1
        for d in shape.reversed() { strides.insert(NSNumber(value: s), at: 0); s *= d }
        arr = try MLMultiArray(dataPointer: ptr!, shape: shape.map { NSNumber(value: $0) },
                               dataType: dt, strides: strides)
    } else {
        arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: dt)
        _ = data.withUnsafeBytes { memcpy(arr.dataPointer, $0.baseAddress!, data.count) }
    }
    feats[name] = MLFeatureValue(multiArray: arr)
}

// warm
let tc0 = DispatchTime.now().uptimeNanoseconds
_ = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats))
let coldMs = Double(DispatchTime.now().uptimeNanoseconds - tc0) / 1e6
var times: [Double] = []
for _ in 0..<burst {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let res = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats))
    let dt = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
    times.append(dt)
    if let y = res.featureValue(for: "y")?.multiArrayValue {
        let n = y.count
        let out = Data(bytes: y.dataPointer, count: n * elemSize(y.dataType))
        try out.write(to: URL(fileURLWithPath: "\(indir)/OUT_y.bin"))
    }
}
times.sort()
let med = times[times.count / 2]
let enc = JSONEncoder()
struct R: Codable { let med_ms: Double; let min_ms: Double; let burst: Int; let wrap: Bool; let cpu: Bool; let cold_ms: Double }
let s = try enc.encode(R(med_ms: med, min_ms: times.first ?? med, burst: burst, wrap: wrap, cpu: cpu, cold_ms: coldMs))
FileHandle.standardOutput.write(s)
