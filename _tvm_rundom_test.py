import os
from argparse import ArgumentParser

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm
import psutil

import tvm
from tvm import relay, tir, cpu
from tvm.relay import parser
from tvm.ir.transform import Sequential
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib import utils

from Autotuning.debug import ModuleRunner, ModuleError
from Autotuning.sequence import RandomRelaySeq, RelaySeq
from Autotuning.util import gen_tensor_value, gen_tensor_value_dict

# Parse arguments
argparser = ArgumentParser()
argparser.add_argument('-p', '--code_path', type=str, help='Code path.')
argparser.add_argument('-s', '--seed', type=int, default=42, help='Random seed.')
argparser.add_argument('-l', '--length', type=int, default=5, help='Length of sequence.')
argparser.add_argument('-o', '--opt_level', type=int, default=4, help='Default optimization level. When use this param, -q and -l are not used.')
args = argparser.parse_args()

def get_current_memory_gb() -> float:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. # / 1024.

AvailablePasses = [
    relay.transform.AlterOpLayout(),    # 0
    relay.transform.AnnotateSpans(),    # 1
    # relay.transform.AnnotateTarget(), 
    relay.transform.BackwardFoldScaleAxis(),    # 2 
    relay.transform.BatchingOps(),      # 3
    relay.transform.CanonicalizeCast(), # 4
    relay.transform.CanonicalizeOps(),  # 5
    # relay.transform.CapturePostDfsIndexInSpans(), 
    # relay.transform.CollagePartition(), 
    relay.transform.CombineParallelBatchMatmul(), 
    relay.transform.CombineParallelConv2D(),  
    relay.transform.CombineParallelDense(),  
    # relay.transform.Conv2dToSparse(),          
    # relay.transform.Conv2dToSparse2(),       
    # relay.transform.ConvertLayout(),           
    relay.transform.DeadCodeElimination(), 
    relay.transform.DefuseOps(),        # 10
    # relay.transform.DenseToSparse(),            
    relay.transform.DynamicToStatic(), 
    relay.transform.EliminateCommonSubexpr(),    
    relay.transform.EtaExpand(False, True),
    relay.transform.EtaExpand(True, False),            
    relay.transform.FakeQuantizationToInteger(),    # 15  
    relay.transform.FastMath(), 
    # relay.transform.FirstOrderGradient(), 
    relay.transform.FlattenAtrousConv(), 
    relay.transform.FoldConstant(),              
    relay.transform.FoldExplicitPadding(),  
    relay.transform.FoldScaleAxis(),            # 20
    relay.transform.ForwardFoldScaleAxis(), 
    relay.transform.FuseOps(0),
    relay.transform.FuseOps(1),
    relay.transform.FuseOps(2), 
    relay.transform.FuseOps(3),     # 25
    relay.transform.FuseOps(4),             
    relay.transform.InferType(), 
    relay.transform.Inline(), 
    # relay.transform.InlineCompilerFunctionsBoundTo(),
    relay.transform.LambdaLift(),   
    # relay.transform.LazyGradientInit(),     # not support fGraph
    relay.transform.Legalize(),     # 30
    relay.transform.MarkCompilerFunctionsAsExtern(),  
    relay.transform.MergeCompilerRegions(), 
    # relay.transform.MergeComposite(),          
    relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols(),  
    # relay.transform.PartialEvaluate(), 
    relay.transform.PartitionGraph(),          
    # relay.transform.PlanDevices(),          
    relay.transform.RemoveUnusedFunctions(),        # 35 
    relay.transform.SimplifyExpr(), 
    # relay.transform.SimplifyFCTranspose(),     
    relay.transform.SimplifyInference(), 
    relay.transform.SplitArgs(-1),                
    relay.transform.ToANormalForm(),    
    relay.transform.ToBasicBlockNormalForm(),   # 40
    relay.transform.ToGraphNormalForm(),
    # relay.transform.ToMixedPrecision(),
]

def gen_random_seq(length:int, rng:Generator):
    idxes = list(rng.choice(len(AvailablePasses), length))
    seq = [AvailablePasses[id] for id in idxes]
    if relay.transform.Legalize() not in seq:
        seq = [relay.transform.Legalize()] + seq
    if relay.transform.SimplifyInference() not in seq:
        seq = [relay.transform.SimplifyInference()] + seq
    return tvm.ir.transform.Sequential(seq), idxes

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
    
# Parse a relay code file
codePath = args.code_path
with open(codePath, 'r') as f:
    code = f.read()
mod = parser.parse(code)

# Generate input parameters
rng = Generator(PCG64(seed=args.seed))
main_fn = mod['main']
inputs = gen_tensor_value_dict(main_fn.params[0:1], rng)
params = gen_tensor_value_dict(main_fn.params[1:], rng)

with tvm.transform.PassContext(opt_level=args.opt_level, disabled_pass = [
    'qnn.Legalize', 'QnnLegalize', 'QnnCanonicalize']) as PC:
    seq, idxes = gen_random_seq(length = args.length, rng=rng)
    try:
        mod = seq(mod)

        disableBefore = DisablePassBefore('SplitArgs')
        PC.override_instruments([disableBefore])
        lib = relay.build(mod, target='llvm', params=params)

    except:
        raise Exception(str(seq))

import pickle
seq_dir = os.path.join(os.path.dirname(args.code_path), 'seq')
if not os.path.exists(seq_dir):
    os.mkdir(seq_dir)
# with open(os.path.join(seq_dir, 'seq.pkl'), 'wb') as f:
#     pickle.dump(seq, f)
with open(os.path.join(seq_dir, 'seq.txt'), 'w') as f:
    f.write(str(idxes))

'''
Memory footprint profiling
'''
temp = utils.tempdir()
path_lib = temp.relpath("deploy_lib.tar")
lib.export_library(path_lib)

mem_s = get_current_memory_gb()     # warm up
mem_s = get_current_memory_gb()
loaded_lib = tvm.runtime.load_module(path_lib)
# with open('./inputs.pkl', 'rb') as f:
#     inputs = pickle.load(f)
gmod = GraphModule(loaded_lib['default'](cpu()))
gmod.run(**inputs)
mem_t = get_current_memory_gb()
print(mem_t - mem_s, 'mb')

# python ./_tvm_rundom_test.py -p=./eval/resnet/resnet18_train.txt -l=10