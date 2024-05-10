import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.backend import Executor
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing import run_infer_type, create_workload

def annot_expr(e):
    """Returns e wrapped with an on_device annotation."""
    return relay.op.annotation.on_device(e, tvm.cpu(), constrain_result=True)

data = tvm.nd.array(np.array([1, 2, 3], dtype="int8"))
const_i8 = relay.const(data, dtype="int8")
op = relay.qnn.op.requantize(
            const_i8,
            input_scale=relay.const(2.0, dtype="float32"),
            input_zero_point=relay.const(1, dtype="int32"),
            output_scale=relay.const(1.0, dtype="float32"),
            output_zero_point=relay.const(1, dtype="int32"),
)
x = relay.var("x", relay.TensorType([3], "int8"))
add = relay.op.add(op, x)
f = relay.Function([x], add)
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
