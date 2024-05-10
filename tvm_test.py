import os
from numpy.random import Generator, PCG64
from typing import List
import tvm
from tvm.relay import parser
from tvm import relay, transform, cpu
from tvm.contrib.graph_executor import GraphModule
from Autotuning.util import simu_mem_from_relay
import psutil
from tvm.contrib import utils
from tvm import transform
import time
import pickle

root_dir = './utils/test/'

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)

def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

def get_current_memory_gb() -> float:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. # / 1024.

def save_relay(mod, filePath):
    with open(filePath, 'w') as f:
        f.write(mod.astext())

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def __init__(self, focus_pass : str = ''):
        self._info = []

        self._focus = focus_pass

    def run_before_pass(self, mod, info):
        self._info.append(info)

        if self._focus and self._focus == str(info.name):
            save_relay(mod, os.path.join(root_dir, 'focus_pass_before.txt'))
            
    def getinfo(self):
        return self._info

    def run_after_pass(self, mod, info):
        if self._focus and self._focus == str(info.name):
            save_relay(mod, os.path.join(root_dir, 'focus_pass_after.txt'))

@tvm.instrument.pass_instrument
class DisablePassAfter:
    """Disable all passes after the specified pass, excluding itself."""

    def __init__(self, passName : str):
        self._should_run = True
        self._mod = None

        self._pass = passName

    def should_run(self, mod, info):
        return self._should_run
    
    def run_after_pass(self, mod, info):
        if str(info.name) == self._pass:
            self._mod = mod
            self._should_run = False

@tvm.instrument.pass_instrument
class DisablePassBefore:
    """Disable all passes before the specified pass, including itself."""

    def __init__(self, passName : str, order : int = 0):
        self._should_run = False
        self._need_change_should = False

        self._pass = passName
        self._target_order = order
        self._current_order = 0

    def should_run(self, mod, info):
        if str(info.name) == 'sequential':
            return True

        if self._need_change_should:
            self._should_run = True
        if str(info.name) == self._pass:
            if self._target_order == self._current_order:
                self._current_order += 1
                self._need_change_should = True
            else:
                self._current_order += 1
        return self._should_run

@tvm.instrument.pass_instrument
class OnlyRunPass:
    """Only run the specified pass."""

    WorkAlongPasses = {
        'PlanDevices' : ['InferType', 'PlanDevicesRewrite', 'PlanDevicesCore'], 
        'FuseOps' : ['InferType']
    }

    def __init__(self, passName : str, order : int = 0) -> None:
        '''
        passName: specified pass to run.
        order: ignore first # specified passes.
        '''
        self._pass = passName
        self._mod = None
        self._target_order = order
        self._current_order = 0
    
    def should_run(self, mod, info):
        if str(info.name) == 'sequential':
            return True
        if str(info.name) == self._pass:
            if self._target_order == self._current_order:
                self._current_order += 1
                return True
            else:
                self._current_order += 1
        if self._pass in self.WorkAlongPasses and str(info.name) in self.WorkAlongPasses[self._pass]:
            return True
        return False

    def run_after_pass(self, mod, info):
        self._mod = mod
    
AvailablePasses = [
    relay.transform.AlterOpLayout(), 
    relay.transform.AnnotateSpans(), 
    # relay.transform.AnnotateTarget(), 
    relay.transform.BackwardFoldScaleAxis(), 
    relay.transform.BatchingOps(), 
    relay.transform.CanonicalizeCast(), 
    relay.transform.CanonicalizeOps(), 
    # relay.transform.CapturePostDfsIndexInSpans(), 
    # relay.transform.CollagePartition(), 
    relay.transform.CombineParallelBatchMatmul(), 
    relay.transform.CombineParallelConv2D(),  
    relay.transform.CombineParallelDense(),  
    # relay.transform.Conv2dToSparse(),          
    # relay.transform.Conv2dToSparse2(),       
    # relay.transform.ConvertLayout(),           
    relay.transform.DeadCodeElimination(), 
    relay.transform.DefuseOps(), 
    # relay.transform.DenseToSparse(),            
    relay.transform.DynamicToStatic(), 
    relay.transform.EliminateCommonSubexpr(),    
    relay.transform.EtaExpand(False, True),
    relay.transform.EtaExpand(True, False),              
    relay.transform.FakeQuantizationToInteger(), 
    relay.transform.FastMath(), 
    # relay.transform.FirstOrderGradient(), 
    relay.transform.FlattenAtrousConv(), 
    relay.transform.FoldConstant(),              
    relay.transform.FoldExplicitPadding(), 
    relay.transform.FoldScaleAxis(), 
    relay.transform.ForwardFoldScaleAxis(), 
    relay.transform.FuseOps(0),
    relay.transform.FuseOps(1),
    relay.transform.FuseOps(2),
    relay.transform.FuseOps(3),
    relay.transform.FuseOps(4),             
    relay.transform.InferType(), 
    relay.transform.Inline(), 
    # relay.transform.InlineCompilerFunctionsBoundTo(),
    relay.transform.LambdaLift(), 
    # relay.transform.LazyGradientInit(),     # not support fGraph
    relay.transform.Legalize(),    
    relay.transform.MarkCompilerFunctionsAsExtern(),  
    relay.transform.MergeCompilerRegions(), 
    # relay.transform.MergeComposite(),          
    relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols(),  
    # relay.transform.PartialEvaluate(), 
    relay.transform.PartitionGraph(),           
    # relay.transform.PlanDevices(),          
    relay.transform.RemoveUnusedFunctions(),    
    relay.transform.SimplifyExpr(), 
    # relay.transform.SimplifyFCTranspose(),     
    relay.transform.SimplifyInference(), 
    relay.transform.SplitArgs(-1),                
    relay.transform.ToANormalForm(),
    relay.transform.ToBasicBlockNormalForm(),
    relay.transform.ToGraphNormalForm(),
    # relay.transform.ToMixedPrecision(),
]

default_seq_0 = tvm.ir.transform.Sequential(
    [
        relay.transform.Legalize(),
        relay.transform.InferType(),
        relay.transform.SimplifyInference(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.FuseOps(0),      # opt level = 0
    ],
)

disabled_pass_4 = ['qnn.Legalize', 'QnnLegalize', 'QnnCanonicalize', 
                 'RemoveUnusedFunctions', 'ToBasicBlockNormalForm', 'Legalize', 
                  'SimplifyInference', 'EliminateCommonSubexpr', 
                 'CombineParallelConv2d', 'CombineParallelDense', 'CombineParallelBatchMatmul', 
                 'FoldConstant', 'FoldScaleAxis', 'BackwardFoldScaleAxis', 'ForwardFoldScaleAxis',
                 'SimplifyExpr', 'CanonicalizeCast', 'CanonicalizeOps', 'FlattenAtrousConv', 
                 'AlterOpLayout', 'FastMath', 'SplitArgs',]  # 

default_seq_4 = tvm.ir.transform.Sequential(
    [
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.transform.Legalize(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),

        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyExpr(),
        
        # relay.transform.ToMixedPrecision(),
        relay.transform.FastMath(),
        # relay.transform.FoldConstant(),
        # relay.transform.SplitArgs(10), # 25~50
        # relay.transform.FuseOps(4),
    ],
)

HOST_TARGET = tvm.target.Target("llvm")
CPU_TARGET = tvm.target.Target("llvm").with_host(HOST_TARGET)
CTXT = tvm.transform.PassContext()
config = tvm.target.make_compilation_config(CTXT, CPU_TARGET)

custom_seq = tvm.ir.transform.Sequential(
    [
        relay.transform.ToGraphNormalForm(),
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.transform.Legalize(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),

        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyExpr(),
        
        relay.transform.FastMath(),
        relay.transform.ToMixedPrecision(),
        # relay.transform.FoldConstant(),
        # relay.transform.SplitArgs(10), # 25~50
        # relay.transform.FuseOps(4),
    ],
)

# idxes = [1, 21, 19, 40, 8, 39, 18, 39, 14, 38, 20, 1, 32, 29, 41, 15, 15, 18, 19, 17]
# idxes = [41]   #
# seq = [AvailablePasses[id] for id in idxes]
# if relay.transform.Legalize() not in seq:
#     seq = [relay.transform.Legalize()] + seq
# if relay.transform.SimplifyInference() not in seq:
#     seq = [relay.transform.SimplifyInference()] + seq
# custom_seq = tvm.ir.transform.Sequential(seq)
# print(seq)

codePath = './eval/transformer/transformer.txt'
# codePath = './eval/resnet/resnet18_train.txt'
# newCodePath = './eval/transformer/transformer_afteropt.txt'

# Parse a relay code file
# codePath = './utils/test/code.txt'
with open(codePath, 'r') as f:
    code = f.read()
mod = parser.parse(code)
# print(simu_mem_from_relay(mod))

# Generate input parameters
rng = Generator(PCG64(seed=42))
main_fn = mod['main']
inputs = gen_tensor_value_dict(main_fn.params[0:1], rng)
params = gen_tensor_value_dict(main_fn.params[1:], rng)

with open('./inputs.pkl', 'wb') as f:
    pickle.dump(inputs, f)
'''
Optimize the relay code with the default sequence
'''
# with transform.PassContext(opt_level=4, instruments=[printIR]) as PC:
#     mod = default_seq_4(mod)
#     # mod = relay.transform.PlanDevices(tvm.target.make_compilation_config(PC, 'llvm', target_host='llvm'))(mod)
#     # mod = relay.transform.FuseOps(4)(mod)
# with open('./utils/test/pass_before.txt', 'w') as f:
#     f.write(str(printIR.getinfo()))

# print(simu_mem_from_relay_test(mod, params))

# Save the optimized relay code to file
# with open(newCodePath, 'w') as f:
#     f.write(mod.astext())

'''
Build and run
'''
# with transform.PassContext(opt_level=4, instruments=[printIR, disableAfter], disabled_pass = [
#     'qnn.Legalize', 'QnnLegalize', 'QnnCanonicalize']): # , disabled_pass=disabled_pass_4
#     try:
#         lib = relay.build(mod, target='llvm', params=params)
#     except:
#         pass
# mod = disableAfter._mod
# with open('./utils/test/pass_before.txt', 'w') as f:
#     f.write(str(printIR.getinfo()))

# printIR._info = []
# with transform.PassContext(opt_level=4, instruments=[printIR, disableBefore]): #, disabled_pass=disabled_pass_4 
    
    # lib = relay.build(mod, target='llvm', params=params)

# with open('./utils/test/pass_after.txt', 'w') as f:
#     f.write(str(printIR.getinfo()))
# with open('./utils/test/code.txt', 'w') as f:
#     f.write(mod.astext())

def run_pass(mod : tvm.ir.module.IRModule, passContext: tvm.ir.transform.PassContext, 
             passName : str, order : int = 0, instruments : List = []):
    runPassInstr = OnlyRunPass(passName, order)
    passContext.override_instruments(instruments + [runPassInstr])
    try:
        _ = relay.build(mod, target='llvm', params=params)
    except:
        pass
    return runPassInstr._mod

printIR = PrintIR(focus_pass='FuseOps')
disableAfter = DisablePassAfter('FastMath')
disableBefore = DisablePassBefore('FastMath')
with transform.PassContext(opt_level=4, instruments=[printIR], disabled_pass = [
    'qnn.Legalize', 'QnnLegalize', 'QnnCanonicalize']) as PC:

    # PC.override_instruments([printIR, disableAfter])
    # try:
    #     lib = relay.build(mod, target='llvm', params=params)
    # except:
    #     pass
    # mod = disableAfter._mod

    # mod = default_seq_4(mod)
    mod = custom_seq(mod)
    
    # mod = relay.transform.InferType()(mod)
    # save_relay(mod, './utils/test/mod_afterinferCustom.txt')
    # mod = relay.transform.InferType()(mod)
    # mod = relay.transform.FoldConstant()(mod)
    # CPU_TARGET = tvm.target.Target("llvm").with_host(tvm.target.Target("llvm"))
    # config = tvm.target.make_compilation_config(PC, CPU_TARGET)
    # mod = relay.transform.InferType()(mod)
    # mod = relay.transform.PlanDevices(config)(mod)
    # mod = relay.transform.InferType()(mod)
    # mod = relay.transform.SplitArgs(-1)(mod)
    # mod = relay.transform.InferType()(mod)
    # mpd = relay.transform.FuseOps(4)(mod)
    # save_relay(mod, './utils/test/mod_afterfoldCustom.txt')
    
    # mod = run_pass(mod, PC, 'FoldConstant', 1, [printIR])
    # mod = run_pass(mod, PC, 'SplitArgs', 0, [printIR])
    # mod = run_pass(mod, PC, 'PlanDevices', 0, [printIR])
    # mod = run_pass(mod, PC, 'FuseOps', 0, [printIR])

    # print(simu_mem_from_relay_test(relay.transform.FuseOps(4)(mod), params), 'mb')

    # mod = relay.transform.ToANormalForm()(mod)
    # mod = parser.parse(mod.astext())
    # mod = tvm.ir.transform.Sequential([
    #     relay.transform.ToMixedPrecision(),
    # ])(mod)
    
    with open('./utils/test/pass_before.txt', 'w') as f:
        f.write(str(printIR.getinfo()))
    
    printIR._info = []
    PC.override_instruments([printIR, disableBefore])
    lib = relay.build(mod, target='llvm', params=params)
    with open('./utils/test/pass_after.txt', 'w') as f:
        f.write(str(printIR.getinfo()))

# with transform.PassContext(opt_level=4, instruments=[printIR], disabled_pass = [
#     'qnn.Legalize', 'QnnLegalize', 'QnnCanonicalize']): #, disabled_pass=disabled_pass_4 
#     lib = relay.build(mod, target='llvm', params=params)
# with open('./utils/test/pass_resnet.txt', 'w') as f:
#     f.write(str(printIR.getinfo()))

temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
path_lib = './utils/test/tmp.tar'
lib.export_library(path_lib)

'''
Memory footprint profiling
'''
time_s = time.time()
mem_s = get_current_memory_gb()     # warm up
mem_s = get_current_memory_gb()
loaded_lib = tvm.runtime.load_module(path_lib)
# with open('./inputs.pkl', 'rb') as f:
#     inputs = pickle.load(f)
gmod = GraphModule(loaded_lib['default'](cpu()))
gmod.run(**inputs)
mem_t = get_current_memory_gb()
time_t = time.time()
print('Profiled memory footprint:', mem_t - mem_s, 'mb')
print('Running time:', time_t - time_s, 's')
# default 276.83mb

'''
Run
'''
# gmod =  GraphModule(lib['default'](cpu()))
# gmod.run(**inputs)


'''
Time evaluation
'''
# gmod.set_input(**inputs)

# num = 10  # number of times we run module for a single measurement
# rep = 3  # number of measurements (we derive std dev from this)
# timer = gmod.module.time_evaluator("run", cpu(), number=num, repeat=rep)

# tcost = timer()
# mean = tcost.mean * 1000    # Average per sample inference time / ms
# print(mean)

'''
Transformer:
TVM: 150MB
Passes:
        No-opt, 2024MB
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.transform.Legalize(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),
        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyExpr(),
        relay.transform.FastMath(),         
        relay.transform.FoldConstant(), 2024MB
        relay.transform.SplitArgs(10),  2024MB
        relay.transform.FuseOps(),      150MB

OPASS(ToMixed): 100MB
Passes:
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.transform.Legalize(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),

        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyExpr(),
        
        relay.transform.FastMath(),
        relay.transform.ToMixedPrecision(),

PyTorch: 460MB
TensorFlow: 350MB

ResNet 18 Train:
TVM: 310MB
OPASS(ToGraph): 200MB
Passes:
        relay.transform.ToGraphNormalForm(),
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.transform.Legalize(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),

        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyExpr(),
        
        relay.transform.FastMath(),
OPASS(ToMixed, ToGraph): 130MB
Passes:
        relay.transform.ToGraphNormalForm(),
        relay.transform.RemoveUnusedFunctions(),
        relay.transform.ToBasicBlockNormalForm(),
        relay.transform.Legalize(),
        relay.transform.SimplifyInference(),
        relay.transform.EliminateCommonSubexpr(),
        relay.transform.CombineParallelConv2D(),
        relay.transform.CombineParallelDense(),
        relay.transform.CombineParallelBatchMatmul(),

        relay.transform.FoldConstant(),
        relay.transform.FoldScaleAxis(),
        relay.transform.SimplifyExpr(),
        relay.transform.CanonicalizeCast(),
        relay.transform.CanonicalizeOps(),
        relay.transform.FlattenAtrousConv(),
        relay.transform.AlterOpLayout(),
        relay.transform.SimplifyExpr(),
        
        relay.transform.FastMath(),
        relay.transform.ToMixedPrecision(),

PyTorch: 150MB
PyTorch: 560MB
'''

'''
sequential 0
RemoveUnusedFunctions 1
ToBasicBlockNormalForm 2
InferType 3
Legalize 4
InferType 5
SimplifyInference 6
InferType 7
EliminateCommonSubexpr 8
InferType 9
CombineParallelConv2d 10
InferType 11
CombineParallelDense 12
InferType 13
CombineParallelBatchMatmul 14
FoldConstant 15
InferType 16
FoldScaleAxis 17
InferType 18
BackwardFoldScaleAxis 19
InferType 20
InferType 21
ForwardFoldScaleAxis 22
InferType 23
FoldConstant 24
InferType 25
InferType 26
SimplifyExpr 27
InferType 28
InferType 29
InferType 30
InferType 31
InferType 32
InferType 33
InferType 34
InferType 35
InferType 36
InferType 37
InferType 38
InferType 39
InferType 40
InferType 41
InferType 42
InferType 43
InferType 44
InferType 45
InferType 46
InferType 47
InferType 48
InferType 49
InferType 50
InferType 51
InferType 52
InferType 53
CanonicalizeCast 54
InferType 55
InferType 56
CanonicalizeOps 57
InferType 58
InferType 59
FlattenAtrousConv 60
InferType 61
InferType 62
InferType 63
AlterOpLayout 64
InferType 65
InferType 66
SimplifyExprPostAlterOp 67
InferType 68
InferType 69
InferType 70
InferType 71
InferType 72
InferType 73
InferType 74
InferType 75
FastMath 76
InferType 77
FoldConstant 78
InferType 79
InferType 80
SplitArgs 81
InferType 82
PlanDevices 83
PlanDevicesRewrite 84
InferType 85
PlanDevicesCore 86
InferType 87
FuseOps 88
InferType 89
InferType 90
InlineGlobals 91
InferType 92
LabelOps 93
InferType 94
AnnotateMemoryScope 95
InferType 96
sequential 97
RelayToTIRTargetHook 98
InferType 99
LowerTE 100
LowerTensorExpr 101
'''