import os
from numpy.random import Generator, PCG64
from typing import List
import tvm
from tvm.relay import parser
from tvm import relay, transform, cpu
from tvm.contrib.graph_executor import GraphModule

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)


def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

default_seq = tvm.ir.transform.Sequential(
    [
        relay.transform.FoldExplicitPadding(),
    ],
)

# Parse a relay code file
with open('./code.txt', 'r') as f:
    code = f.read()
mod = parser.parse(code)

# Generate input parameters
rng = Generator(PCG64(seed=42))
main_fn = mod['main']
inputs = gen_tensor_value_dict(main_fn.params[0:1], rng)
params = gen_tensor_value_dict(main_fn.params[1:], rng)

# Optimize the relay code with the default sequence
with transform.PassContext(opt_level=4):
    mod = default_seq(mod)

# Save the optimized relay code to file
with open('./new_code.txt', 'w') as f:
    f.write(mod.astext())



# gmod.set_input(**inputs)

# num = 10  # number of times we run module for a single measurement
# rep = 3  # number of measurements (we derive std dev from this)
# timer = gmod.module.time_evaluator("run", cpu(), number=num, repeat=rep)

# tcost = timer()
# mean = tcost.mean * 1000    # Average per sample inference time / ms
# print(mean)
