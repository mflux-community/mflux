// ANE matmul benchmark. Usage: anebench MODEL.mlpackage [--units ane|all|gpu|cpu] [--iters N]
// Model name: <v1|v2>_<T>x<O>x<I>.mlpackage
import CoreML
import Foundation

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }
func rssMB() -> Double {
    var info = mach_task_basic_info(); var c = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / 4)
    let r = withUnsafeMutablePointer(to: &info) { $0.withMemoryRebound(to: integer_t.self, capacity: 1) { task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &c) } }
    return r == KERN_SUCCESS ? Double(info.resident_size) / 1e6 : -1
}
let a = CommandLine.arguments
guard a.count >= 2 else { print("usage"); exit(2) }
var units = MLComputeUnits.cpuAndNeuralEngine, iters = 50, i = 2
while i < a.count {
    if a[i] == "--units" { switch a[i+1] { case "ane": units = .cpuAndNeuralEngine; case "gpu": units = .cpuAndGPU; case "cpu": units = .cpuOnly; default: units = .all }; i += 2 }
    else if a[i] == "--iters" { iters = Int(a[i+1])!; i += 2 } else { i += 1 }
}
let path = a[1]
let base = (path as NSString).lastPathComponent.replacingOccurrences(of: ".mlpackage", with: "")
let parts = base.split(separator: "_"); let variant = String(parts[0])
let d = parts[1].split(separator: "x").map { Int($0)! }; let (T, O, I) = (d[0], d[1], d[2])

let rss0 = rssMB()
var t0 = now()
let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: path))
let tCompile = now() - t0
let cfg = MLModelConfiguration(); cfg.computeUnits = units
t0 = now(); let model = try MLModel(contentsOf: compiled, configuration: cfg); let tLoad = now() - t0

func arr(_ shape: [Int], _ type: MLMultiArrayDataType) -> MLMultiArray {
    let m = try! MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: type)
    let n = shape.reduce(1, *)
    if type == .float16 { let p = m.dataPointer.bindMemory(to: Float16.self, capacity: n); for k in 0..<n { p[k] = Float16(Float.random(in: -1...1)) } }
    else { let p = m.dataPointer.bindMemory(to: Int8.self, capacity: n); for k in 0..<n { p[k] = Int8.random(in: -127...127) } }
    return m
}
let cl = variant.hasPrefix("c")
let pre = variant.hasSuffix("p")
var feats: [String: MLFeatureValue] = ["x": MLFeatureValue(multiArray: arr(cl ? [1, I, 1, T] : [T, I], pre ? .int8 : .float16))]
if pre { let s = arr([1, 1, 1, T], .float16); let p = s.dataPointer.bindMemory(to: Float16.self, capacity: T); for k in 0..<T { p[k] = 0.02 }; feats["tok"] = MLFeatureValue(multiArray: s) }
if variant == "v1" {
    feats["w"] = MLFeatureValue(multiArray: arr([O, I], .int8))
    let s = arr([O], .float16); let p = s.dataPointer.bindMemory(to: Float16.self, capacity: O); for k in 0..<O { p[k] = 0.005 }
    feats["s"] = MLFeatureValue(multiArray: s)
}
let provider = try MLDictionaryFeatureProvider(dictionary: feats)
let opts = MLPredictionOptions()
let out = try MLMultiArray(shape: (cl ? [1, O, 1, T] : [T, O]).map { NSNumber(value: $0) }, dataType: .float16)
opts.outputBackings = ["y": out]

t0 = now(); _ = try model.prediction(from: provider, options: opts); let tFirst = now() - t0
for _ in 0..<10 { _ = try model.prediction(from: provider, options: opts) }
var ts: [Double] = []
var w0 = 0.0, w1 = 0.0
if let k = a.firstIndex(of: "--window") { w0 = Double(a[k+1])!; w1 = Double(a[k+2])! }
if w1 > 0 {   // windowed mode: loop until w1, keep calls that start inside [w0, w1]
    while Date().timeIntervalSince1970 < w1 { let st = Date().timeIntervalSince1970; let s = now(); _ = try model.prediction(from: provider, options: opts); if st >= w0 { ts.append(now() - s) } }
} else { for _ in 0..<iters { let s = now(); _ = try model.prediction(from: provider, options: opts); ts.append(now() - s) } }
guard !ts.isEmpty else { FileHandle.standardError.write("measurement window missed: no samples collected\n".data(using: .utf8)!); exit(3) }
ts.sort(); let med = ts[ts.count / 2], p90 = ts[Int(Double(ts.count) * 0.9)]
let tflops = 2.0 * Double(T) * Double(I) * Double(O) / med / 1e12
let rss1 = rssMB()
print("{\"model\":\"\(base)\",\"units\":\"\(units.rawValue)\",\"compile_s\":\(tCompile),\"load_s\":\(tLoad),\"first_call_s\":\(tFirst),\"median_ms\":\(med*1e3),\"p90_ms\":\(p90*1e3),\"tflops\":\(tflops),\"rss_delta_mb\":\(rss1-rss0)}")
