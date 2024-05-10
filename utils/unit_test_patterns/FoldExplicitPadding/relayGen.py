import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass

import numpy as np

x = relay.var("x", shape=(1, 1, 2, 2), dtype="int8")
weight = relay.var("weight", shape=(1, 1, 2, 2), dtype="int8")
# Pad value and input zp are not equal
pad_value = 1
input_zero_point = 0
pad = relay.nn.pad(x, [[0, 0], [0, 0], [1, 1], [1, 1]], pad_value=pad_value)
op = relay.qnn.op.conv2d(
            pad,
            weight,
            relay.const(input_zero_point, "int32"),
            relay.const(0, "int32"),
            relay.const(1, "float32"),
            relay.const(1, "float32"),
            channels=1,
            kernel_size=(2, 2),
            padding=(0, 0),
)



mod = tvm.IRModule.from_expr(op)
            

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
