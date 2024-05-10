import numpy as np
import tvm
from tvm import te

from tvm import relay
from tvm.relay.op import register_alter_op_layout
from tvm.relay import transform, analysis

x = relay.var("x", shape=(1, 16))
y1 = relay.nn.relu(x)
y2 = relay.nn.relu(x)
y1 = relay.add(y1, relay.const(1.0, "float32"))
y2 = relay.add(y2, relay.const(1.0, "float32"))
c0 = relay.const(np.ones((1, 16)), "float32")
y1 = relay.concatenate([y1, c0], axis=0)
y2 = relay.concatenate([y2, c0], axis=0)
y = relay.add(y1, y2)
f = relay.Function([x], y)
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
