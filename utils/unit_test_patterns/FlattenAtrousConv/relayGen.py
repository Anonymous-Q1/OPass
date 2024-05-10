import numpy as np
import pytest
import tvm
import tvm.testing
from tvm import relay
from tvm.contrib import graph_executor



shape_x = [1, 5, 5, 4]
shape_w = [3, 3, 4, 1]

x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

weight = relay.const(w_np)
data = relay.var("data", shape=shape_x, dtype="float32")
op1 = relay.nn.space_to_batch_nd(data, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
op2 = relay.nn.conv2d(
        op1,
        weight,
        padding=[0, 0, 0, 0],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
)
expr = relay.nn.batch_to_space_nd(op2, block_shape=[2, 2], crops=[[0, 1], [0, 1]])
            
mod = tvm.IRModule.from_expr(expr)


with open('./code.txt', 'w') as f:
    f.write(mod.astext())
