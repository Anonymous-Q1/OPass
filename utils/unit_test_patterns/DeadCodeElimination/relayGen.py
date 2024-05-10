import tvm
import tvm.testing
from tvm.relay import Function, transform
from tvm.relay.testing import inception_v3
import pytest

cpu_scope = tvm.target.VirtualDevice(tvm.cpu(), tvm.target.Target("llvm"))
metatable = {"VirtualDevice": [cpu_scope]}
core = tvm.IRModule()
core.import_from_std("core.rly")
    
n, c, h, w = 1, 16, 64, 64
data = relay.var("data", relay.TensorType((n, c, h, w)))
inp = relay.var("inp", relay.TensorType((n, c * h * w)))
weight_T = relay.const(np.random.random((n, c * h * w)), dtype="float32")
bias = relay.const(np.random.random((n,)), dtype="float32")
conv_w = relay.const(np.random.random((16, 16, 3, 3)), dtype="float32")

dense_o = relay.nn.dense(inp, weight_T)
linear_o = relay.nn.bias_add(dense_o, bias)
conv2d_o = relay.nn.conv2d(data, conv_w, kernel_size=(3, 3), padding=(1, 1), channels=16)
result = relay.Tuple((linear_o, conv2d_o))

mod = tvm.IRModule.from_expr(result)


with open('./code.txt', 'w') as f:
    f.write(mod.astext())
