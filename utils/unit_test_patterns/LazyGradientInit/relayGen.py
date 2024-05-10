import numpy as np

import tvm
from tvm import relay
from tvm.relay import create_executor, transform
from tvm.relay.testing import rand, run_infer_type
import tvm.testing
from tvm.testing import assert_allclose

def before():
    mod = tvm.IRModule()

    shape = (10, 10)
    dtype = "float32"
    t = relay.TensorType(shape, dtype)

    x = relay.var("x", t)
    y = relay.Function([x], x + relay.ones_like(x))

    mod["main"] = y
    return mod
    
mod = before()

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
