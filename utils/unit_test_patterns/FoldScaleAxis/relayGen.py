import numpy as np

import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import create_workload
from tvm.relay.build_module import bind_params_by_name

def initializer(_, param):
    param = np.zeros(param.shape)


def _get_positive_scale(size):
    return np.random.uniform(0.5, 1, size=size).astype("float32")

shape = (2,2,10,10,10,16)
in_channels = 32
channels = 64
blocking = (16,16)


x = relay.var("x", shape=shape)
conv_weight = relay.var("weight")
out_bias = relay.var("out_bias", shape=(channels,))
out_scale = relay.const(_get_positive_scale((channels,)))

args = [x, conv_weight, out_bias]
out_bias = relay.reshape(out_bias, (1, channels // blocking[1], 1, 1, 1, blocking[1]))

y = relay.nn.conv3d(
            x,
            conv_weight,
            channels=channels,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1),
            data_layout="NCDHW{}c".format(blocking[0]) if blocking else "NCDHW",
            kernel_layout="OIDHW1i{}o".format(blocking[1]) if blocking else "OIDHW",
)
y = relay.add(y, out_bias)
y = relay.nn.relu(y)

out_scale = relay.reshape(out_scale, (1, channels // blocking[1], 1, 1, 1, blocking[1]))
y = relay.multiply(y, out_scale)
f = relay.Function(args, y)
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
