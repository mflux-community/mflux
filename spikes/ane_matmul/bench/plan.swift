// Print the preferred compute device per op. Usage: aneplan MODEL.mlpackage
import CoreML
import Foundation
let url = try MLModel.compileModel(at: URL(fileURLWithPath: CommandLine.arguments[1]))
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let plan = try await MLComputePlan.load(contentsOf: url, configuration: cfg)
guard case .program(let prog) = plan.modelStructure, let fn = prog.functions["main"] else { print("no program"); exit(1) }
for op in fn.block.operations {
    let u = plan.deviceUsage(for: op)
    func n(_ d: MLComputeDevice?) -> String { switch d { case .neuralEngine?: return "ANE"; case .gpu?: return "GPU"; case .cpu?: return "CPU"; default: return "?" } }
    print("\(op.operatorName)\t\(n(u?.preferred))")
}
