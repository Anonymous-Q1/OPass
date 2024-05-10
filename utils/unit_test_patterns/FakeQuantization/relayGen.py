import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.transform import fake_quantization_to_integer



shape = [5, 10]
x_ = relay.var("x", shape=shape, dtype="int8")

is_sorted = lambda a: np.all(a[:-1] <= a[1:])

scale = 0.1
x = relay.qnn.op.dequantize(x_, relay.const(scale), relay.const(0))
op = relay.op.nn.softmax(x, axis=1)
op = relay.qnn.op.quantize(
            op, relay.const(1.0 / 256.0), relay.const(-128), out_dtype="int8"
)


mod = tvm.IRModule.from_expr(op)
            



with open('./code.txt', 'w') as f:
    f.write(mod.astext())
