import numpy as np
import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload
import tvm.topi.testing
import tvm.testing

x = relay.var("x", relay.TensorType((2, 2), "int64"))
cond = relay.const(1)
iff = relay.If(cond, relay.reshape(x, [1, 4]), relay.reshape(x, (4, 1)))


f = relay.Function([x], iff)
            
mod = tvm.IRModule.from_expr(f)


with open('./code.txt', 'w') as f:
    f.write(mod.astext())
