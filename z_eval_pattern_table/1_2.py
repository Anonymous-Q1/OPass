# To construct a pattern table and corresponding seq impact
import os
import numpy as np
import tvm
from tvm import relay
from tvm.relay import transform
from Autotuning.util import viz2file, simu_mem_from_relay

root = '/home/nie/RelayOpt/eval/pattern_table/'

'''
Construct pattern
'''
shape_x = [1, 5, 5, 4]
shape_w = [3, 3, 4, 1]

x_np = np.random.randint(-128, 127, size=shape_x, dtype="int8").astype("float32")
w_np = np.random.randint(-128, 127, size=shape_w, dtype="int8").astype("float32")

weight = relay.const(w_np)
data = relay.var("data", shape=shape_x, dtype="float32")

op0 = relay.op.add(data, relay.const(1.0))
op0 = relay.nn.conv2d(
    op0,
    weight,
    padding=[2, 2, 2, 2],
    dilation=[2, 2],
    groups=4,
    channels=4,
    kernel_size=[3, 3],
    data_layout="NHWC",
    kernel_layout="HWOI",
)
op0 = relay.op.add(op0, relay.const(-1.0))

op1 = relay.op.add(data, relay.const(1.0))
op1 = relay.nn.space_to_batch_nd(op1, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
op1 = relay.nn.conv2d(
    op1,
    weight,
    padding=[0, 0, 0, 0],
    groups=4,
    channels=4,
    kernel_size=[3, 3],
    data_layout="NHWC",
    kernel_layout="HWOI",
)
op1 = relay.nn.batch_to_space_nd(op1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])
op1 = relay.op.add(op1, relay.const(-1.0))

z = relay.add(op0, op1)
func = relay.Function([data], z)
before = tvm.IRModule.from_expr(func)
print(before)

'''
Visualize
'''
case_path = os.path.join(root, '1_2.txt')
with open(case_path, 'w') as f:
    f.write(before.astext())
viz2file(case_path)

'''
Origin memory
'''
before = transform.InferType()(before)
origin_mem = simu_mem_from_relay(before)
print('Origin:', origin_mem, 'mem')

'''
Optimized memory
'''
default = transform.EliminateCommonSubexpr()(before)
default = transform.FlattenAtrousConv()(default)
default_mem = simu_mem_from_relay(default)

opass = transform.FlattenAtrousConv()(before)
opass = transform.EliminateCommonSubexpr()(opass)
print(opass)
opass_mem = simu_mem_from_relay(opass)

print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)