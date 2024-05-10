import numpy as np
import tvm
from tvm import te

from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import transform, analysis
from tvm.relay.testing.temp_op_attr import TempOpAttr

def before():
        x = relay.var("x", shape=(1, 64, 56, 56))
        y = relay.var("y", shape=(1, 64, 56, 20))
        z = relay.var("z", shape=(1, 64, 56, 10))
        func = relay.concatenate([x, y, z], axis=3)
        func = relay.Function([x, y, z], func)
        return func

f = before()
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
