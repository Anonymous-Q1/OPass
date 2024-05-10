import os
import tvm
from tvm import relay, parser



case_dir = '/home/nie/RelayOpt/GenCoG/out/run-20230509-014533/2/'
with open(os.path.join(case_dir, 'code.txt')) as f:
    mod = relay.parse(f.read())

with tvm.transform.PassContext(opt_level=5, config={'relay.collage.tvm_max_depth':5}) as PC:
    seq = tvm.ir.transform.Sequential(
        [
            # relay.transform.SimplifyInference(),
            # relay.transform.FuseOps(),
            # relay.transform.FoldConstant(),
            # relay.transform.FoldScaleAxis(),
            # relay.transform.AlterOpLayout(),
            # relay.transform.CanonicalizeOps(),
            # relay.transform.CanonicalizeCast(),
            # relay.transform.EliminateCommonSubexpr(),
            # relay.transform.CombineParallelConv2D(),
            # relay.transform.CombineParallelDense(),
            # relay.transform.CombineParallelBatchMatmul(),
            # relay.transform.FastMath(),

            # relay.transform.DivToMul(),

            # relay.transform.AlterOpLayout(),
            # relay.transform.AnnotateSpans(),
            # relay.transform.AnnotateTarget('llvm', True),     # Note: maybe all graphs can trigger this pass
            # relay.transform.AnnotateTarget('llvm', False),
            # relay.transform.BackwardFoldScaleAxis(),
            # relay.transform.BatchingOps(),                    # Note: this pass is just a combination of three pass
            # relay.transform.CanonicalizeOps(),
            # relay.transform.CapturePostDfsIndexInSpans(),
            # relay.transform.CollagePartition(config=tvm.target.make_compilation_config(PC, tvm.target.Target("llvm"))),     # Alarm: Need CUDA!! with tvm.transform.PassContext(opt_level=5, config={'relay.collage.tvm_max_depth':5}) as PC:
            # relay.transform.CombineParallelBatchMatmul(3),
            # relay.transform.CombineParallelConv2D(3),
            # relay.transform.CombineParallelDense(3, True),
            relay.transform.Conv2dToSparse2('NHWC', 3, blocksize=[3,3], sparsity_threshold=0.5),
            # relay.transform.FakeQuantizationToInteger(),
        ],
        # required=['DeadCodeElimination'],
    )
    mod = seq(mod)

with open(os.path.join(case_dir, 'code1.txt'), 'w') as f:
    f.write(mod.astext())
