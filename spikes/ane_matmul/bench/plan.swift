// Print the preferred compute device per op. Usage: aneplan MODEL.mlpackage [--function NAME]
import CoreML
import Foundation
var fnName: String? = nil
var modelPath: String? = nil
var i = 1
while i < CommandLine.arguments.count {
    if CommandLine.arguments[i] == "--function" { fnName = CommandLine.arguments[i+1]; i += 2 }
    else { modelPath = CommandLine.arguments[i]; i += 1 }
}
guard let path = modelPath else { print("usage: aneplan MODEL.mlpackage [--function NAME]"); exit(2) }
let url: URL
if (path as NSString).pathExtension == "mlmodelc" {
    url = URL(fileURLWithPath: path)
} else {
    url = try MLModel.compileModel(at: URL(fileURLWithPath: path))
}
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
if let fn = fnName { cfg.functionName = fn }
let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
guard case .program(let prog) = plan.modelStructure,
      let fn = prog.functions[fnName ?? "main"] else { print("no program"); exit(1) }
for op in fn.block.operations {
    let u = plan.deviceUsage(for: op)
    func n(_ d: MLComputeDevice?) -> String { switch d { case .neuralEngine?: return "ANE"; case .gpu?: return "GPU"; case .cpu?: return "CPU"; default: return "?" } }
    print("\(op.operatorName)\t\(n(u?.preferred))")
}
