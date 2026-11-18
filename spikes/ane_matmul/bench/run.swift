// Run a model once on raw binary inputs. Usage: anerun MODEL.mlpackage OUT.bin IN_x.bin [IN_tok.bin]
// Input order: x, then tok. Dtypes come from the model description. Output is fp16 [O, T] row-major (channels-first model).
import CoreML
import Foundation
let a = CommandLine.arguments
let url = try MLModel.compileModel(at: URL(fileURLWithPath: a[1]))
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let model = try MLModel(contentsOf: url, configuration: cfg)
var feats: [String: MLFeatureValue] = [:]
let names = ["x", "tok"]
for (k, n) in names.enumerated() where a.count > 3 + k {
    let d = model.modelDescription.inputDescriptionsByName[n]!.multiArrayConstraint!
    let m = try MLMultiArray(shape: d.shape, dataType: d.dataType)
    let data = try Data(contentsOf: URL(fileURLWithPath: a[3 + k]))
    let esize = d.dataType == .float16 ? 2 : 1
    let expected = d.shape.map { $0.intValue }.reduce(1, *) * esize
    guard data.count == expected else {
        FileHandle.standardError.write("input \(n): file has \(data.count) bytes, model expects \(expected)\n".data(using: .utf8)!)
        exit(2)
    }
    data.withUnsafeBytes { src in memcpy(m.dataPointer, src.baseAddress!, data.count) }
    feats[n] = MLFeatureValue(multiArray: m)
}
let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: feats)).featureValue(for: "y")!.multiArrayValue!
let sh = out.shape.map { $0.intValue }, st = out.strides.map { $0.intValue }
let C = sh[1], T = sh[3]
var buf = [Float16](repeating: 0, count: C * T)
let p = out.dataPointer.bindMemory(to: Float16.self, capacity: out.count)
for c in 0..<C { for t in 0..<T { buf[c * T + t] = p[c * st[1] + t * st[3]] } }
try Data(bytes: buf, count: buf.count * 2).write(to: URL(fileURLWithPath: a[2]))
