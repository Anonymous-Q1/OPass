# To construct a pattern table and corresponding seq impact
import os
import tvm
from tvm import relay
from tvm.relay import transform
from Autotuning.util import viz2file, simu_mem_from_relay

root = '/home/nie/RelayOpt/eval/pattern_table/'

'''
Construct pattern
'''
x = relay.var("x", shape=[1, 5, 5, 4], dtype="float32")
w = relay.ones([3, 3, 4, 1], dtype='float32')

# x1 = relay.transpose(x, axes=[0, 1, 2, 3])
x1 = relay.nn.conv2d(
        x,
        w,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
# x1 = relay.layout_transform(x1, "NCHW", "NCHW")

y1 = relay.nn.space_to_batch_nd(x, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
y1 = relay.nn.conv2d(
    y1,
    w,
    padding=[0, 0, 0, 0],
    groups=4,
    channels=4,
    kernel_size=[3, 3],
    data_layout="NHWC",
    kernel_layout="HWOI",
)
y1 = relay.nn.batch_to_space_nd(y1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

z = relay.add(x1, y1)
func = relay.Function([x], z)
before = tvm.IRModule.from_expr(func)
print(before)

'''
Visualize
'''
case_path = os.path.join(root, '1.txt')
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
opass_mem = simu_mem_from_relay(opass)

print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)