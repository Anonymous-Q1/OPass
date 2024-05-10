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

# y1 = relay.nn.space_to_batch_nd(x, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
# y1 = relay.nn.conv2d(
#     x,
#     w,
#     padding=[0, 0, 0, 0],
#     groups=4,
#     channels=4,
#     kernel_size=[3, 3],
#     data_layout="NHWC",
#     kernel_layout="HWOI",
# )
# y1 = relay.nn.batch_to_space_nd(y1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

y1 = relay.nn.conv2d(
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

z = relay.add(x1, y1)
func = relay.Function([x], z)
before = tvm.IRModule.from_expr(func)
before = transform.InferType()(before)
before = transform.FuseOps()(before)
print(before)

'''
For test: load mod from file
'''
# file_path  = os.path.join(root, 'test/code.txt')
# with open(file_path, 'r') as f:
#     before = relay.parse(f.read())
# before = transform.DynamicToStatic()(before)

'''
Visualize
'''
case_dir = os.path.join(root, '5')
os.system(f'rm -rf {case_dir}')
os.mkdir(case_dir)
case_path = os.path.join(case_dir, 'code.txt')
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
default = transform.FuseOps(4)(default)
# default = transform.ToMixedPrecision()(default)
# default = transform.ToMixedPrecision()(default)
default_mem = simu_mem_from_relay(default)
# print(default)

# opass = transform.SimplifyExpr()(before)
opass = transform.DefuseOps()(before)
opass = transform.EliminateCommonSubexpr()(opass)
opass = transform.FuseOps(4)(opass)
opass_mem = simu_mem_from_relay(opass)
print(opass)
# 
print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)

# tmp_path = os.path.join(case_dir, 'tmp.txt')
# with open(tmp_path, 'w') as f:
#     f.write(default.astext())
# viz2file(tmp_path)