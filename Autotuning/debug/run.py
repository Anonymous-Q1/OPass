import os
import psutil
from enum import IntEnum, auto
from typing import Dict, Optional, List

import numpy as np
from numpy.random import Generator
import tvm
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.ir.transform import Sequential
from tvm.contrib.graph_executor import GraphModule
from tvm.contrib import utils
from ..util import simu_mem_from_relay, default_seq_dict
from ..sequence import RelaySeq

class ErrorKind(IntEnum):
    PARSE = auto()
    COMPILE = auto()
    RUN = auto()
    COMPUTE = auto()
    TIMING = auto()
    PROFILING = auto()
    SIMU = auto()

TensorDict = Dict[str, np.ndarray]

class ModuleError(Exception):
    def __init__(self, kind: ErrorKind, code: str, err: str, opt_level: int,
                 inputs: Optional[TensorDict] = None, params: Optional[TensorDict] = None,
                 seq: Optional[RelaySeq] = None):
        self.kind_ = kind
        self.code_ = code
        self.err_ = err
        self.opt_level_ = opt_level
        self.inputs_ = inputs
        self.params_ = params
        self.seq_ = seq

    def report(self, path: str):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, self.kind_.name), 'w'):
            pass
        with open(os.path.join(path, 'code.txt'), 'w') as f:
            f.write(self.code_)
        with open(os.path.join(path, 'error.txt'), 'w') as f:
            f.write(f'opt_level={self.opt_level_}\n')
            f.write(self.err_)
        if self.inputs_ is not None:
            np.savez(os.path.join(path, 'inputs.npz'), **self.inputs_)
        if self.params_ is not None:
            np.savez(os.path.join(path, 'params.npz'), **self.params_)
        if self.seq_ is not None:
            # with open(os.path.join(path, 'seq.txt'), 'w') as f:
            #     f.write('\n'.join([str(p) for p in self.seq_.passes]))
            self.seq_.save(os.path.join(path, 'seq.txt'))

class ModuleRunner:
    def __init__(self, rng: Generator):
        self._rng = rng

    def run(self, code: str, seq: RelaySeq, opt_level: Optional[int] = 4):
        # Parse module
        try:
            mod = relay.parse(code)
        except TVMError as err:
            raise ModuleError(ErrorKind.PARSE, code, str(err), 0)

        # Generate input parameters
        main_fn = mod['main']
        inputs = gen_tensor_value_dict(main_fn.params[0:1], self._rng)
        params = gen_tensor_value_dict(main_fn.params[1:], self._rng)

        # Build and run unoptimized module as reference
        try:
            gmod = build_mod(mod, 0, params=params)
        except Exception as err:
            raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), 0)
        try:
            ref_outputs = run_gmod(gmod, inputs)
        except Exception as err:
            raise ModuleError(ErrorKind.RUN, mod.astext(), str(err), 0)

        # Build and run the module with specified pass sequence
        try:
            gmod = build_mod(mod, opt_level, params=params, seq=seq)
        except Exception as err:
            raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), opt_level, seq=seq)
        try:
            outputs = run_gmod(gmod, inputs)
        except Exception as err:
            raise ModuleError(ErrorKind.RUN, mod.astext(), str(err), opt_level, seq=seq)
        for i, (o, ro) in enumerate(zip(outputs, ref_outputs)):
            if not np.allclose(o, ro, rtol=1e-3, atol=1e-2, equal_nan=True):
                msg = f'Computation error in output tensor {i}:\n' \
                        f'Expect:\n' \
                        f'{np.array_repr(ro)}\n' \
                        f'Actual:\n' \
                        f'{np.array_repr(o)}'
                raise ModuleError(ErrorKind.COMPUTE, mod.astext(), msg, opt_level,
                                    inputs=inputs, params=params, seq=seq)
        
        return outputs

    def eval_time(self, code: str, seq: RelaySeq, opt_level: Optional[int] = 4):
        # Parse module
        try:
            mod = relay.parse(code)
        except TVMError as err:
            raise ModuleError(ErrorKind.PARSE, code, str(err), 0)

        # Generate input parameters
        main_fn = mod['main']
        inputs = gen_tensor_value_dict(main_fn.params[0:1], self._rng)
        params = gen_tensor_value_dict(main_fn.params[1:], self._rng)

        # Build and run the module with specified pass sequence
        try:
            gmod = build_mod(mod, opt_level, params=params, seq=seq)
        except Exception as err:
            raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), opt_level, seq=seq)

        # Evaluate execution time of the module with specified pass sequence
        try:
            exec_time = time_gmod(gmod, inputs)
        except:
            raise ModuleError(ErrorKind.TIMING, mod.astext(), str(err), opt_level, seq=seq)

        return exec_time
    
    def eval_mem(self, code: str, seq: RelaySeq, opt_level: Optional[int] = 4):
        # Parse module
        try:
            mod = relay.parse(code)
        except TVMError as err:
            raise ModuleError(ErrorKind.PARSE, code, str(err), 0)

        # Generate input parameters
        main_fn = mod['main']
        inputs = gen_tensor_value_dict(main_fn.params[0:1], self._rng)
        params = gen_tensor_value_dict(main_fn.params[1:], self._rng)

        # Build the module with specified pass sequence
        try:
            if seq is not None:
                with transform.PassContext(opt_level=opt_level):
                    mod = seq.seq(mod)
                with transform.PassContext(opt_level=0):
                    lib = relay.build(mod, target='llvm', params=params)
            else:
                with transform.PassContext(opt_level=opt_level):
                    lib = relay.build(mod, target='llvm', params=params)
        except Exception as err:
            raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), opt_level, seq=seq)
        
        # Save built module to file
        temp = utils.tempdir()
        path_lib = temp.relpath("deploy_lib.tar")
        lib.export_library(path_lib)
            
        # Run the module and evaluate the memory footprint
        try:
            memory = mem_gmod(path_lib, inputs)
            # memory = mem_gmod_multi(path_lib, inputs, repeat=10, warmup=4)
        except Exception as err:
            raise ModuleError(ErrorKind.PROFILING, mod.astext(), str(err), opt_level, seq=seq)
        
        try:
            if seq is None:
                with transform.PassContext(opt_level=opt_level):
                    mod = default_seq_dict[opt_level](mod)
            mod = relay.parse(mod.astext())
            sim_mem = simu_mem_from_relay(mod, params)
        except Exception as err:
            raise ModuleError(ErrorKind.SIMU, mod.astext(), str(err), opt_level, seq=seq)
        
        return memory, sim_mem
    
    def opt(self, code: str, seq: RelaySeq, opt_level: Optional[int] = 4, save_path: Optional[str] = None):
        # Parse module
        try:
            mod = relay.parse(code)
        except TVMError as err:
            raise ModuleError(ErrorKind.PARSE, code, str(err), 0)

        try:
            if seq is not None:
                with transform.PassContext(opt_level=opt_level):
                    mod = seq.seq(mod)
            else:
                with transform.PassContext(opt_level=opt_level):
                    mod = default_seq_dict[opt_level](mod)
        except Exception as err:
            raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), opt_level, seq=seq)
        
        if save_path is not None:
            with open(save_path, 'w') as f:
                f.write(mod.astext())

        return mod

def get_current_memory_gb() -> float:
    # 获取当前进程内存占用。
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    return info.uss / 1024. / 1024. # / 1024.

def get_current_sys_memory() -> float:
    mem = psutil.virtual_memory()
    return mem.used / 1024. / 1024.

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)


def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}


def build_mod(mod: IRModule, opt_level: int, params: Optional[TensorDict] = None, seq: Optional[RelaySeq] = None):
    if seq is not None:
        with transform.PassContext(opt_level=opt_level):
            mod = seq.seq(mod)
        with transform.PassContext(opt_level=0):
            lib = relay.build(mod, target='llvm', params=params)
    else:
        with transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target='llvm', params=params)
    return GraphModule(lib['default'](cpu()))


def run_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray]) -> List[np.ndarray]:
    gmod.run(**inputs)
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]

def time_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray], number: Optional[int] = 10, repeat: Optional[int] = 3) -> float:
    gmod.set_input(**inputs)

    num = number  # number of times we run module for a single measurement
    rep = repeat  # number of measurements (we derive std dev from this)
    timer = gmod.module.time_evaluator("run", cpu(), number=num, repeat=rep)

    tcost = timer()
    mean = tcost.mean * 1000    # Average per sample inference time / ms
    return mean

# from memory_profiler import profile
# @profile
def mem_gmod(path_lib: str, inputs: Dict[str, np.ndarray]) -> float:
    # Run and profile memory footprint
    mem_s = get_current_memory_gb()     # warm up and avoid the memory used by function get_current_memory_gb itself.
    mem_s = get_current_memory_gb()
    loaded_lib = tvm.runtime.load_module(path_lib)
    gmod = GraphModule(loaded_lib['default'](cpu()))
    gmod.run(**inputs)
    mem_t = get_current_memory_gb()
    # print(mem_t - mem_s)
    return mem_t-mem_s

def mem_gmod_multi(path_lib:str, inputs: Dict[str, np.ndarray], repeat: Optional[int] = 5, warmup: Optional[int] = 5) -> float:
    # warm up
    for _ in range(warmup):
        _ = mem_gmod(path_lib, inputs)
    memory = 0
    for _ in range(repeat):
        memory += mem_gmod(path_lib, inputs)
    memory /= repeat
    return memory