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

x1 = relay.reshape(x, newshape=(1, 20, -1))
x1 = relay.reshape(x1, newshape=(1, 5, 5, 4))
x1 = relay.nn.conv2d(
        x1,
        w,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
x1 = relay.reshape(x1, newshape=(1, 20, -1))
x1 = relay.reshape(x1, newshape=(1, 5, 5, 4))
# x1 = relay.reverse_reshape(x1, newshape=(32, 0, -1))

y1 = relay.reshape(x, newshape=(1, 5, 5, 4))
y1 = relay.nn.conv2d(
        y1,
        w,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
y1 = relay.reshape(y1, newshape=(1, 5, 5, 4))

z = relay.add(x1, y1)
func = relay.Function([x], z)
before = tvm.IRModule.from_expr(func)
print(before)

'''
Visualize
'''
case_dir = os.path.join(root, '2_1')
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
default = before
default = transform.EliminateCommonSubexpr()(default)
default = transform.SimplifyExpr()(default)
default_mem = simu_mem_from_relay(default)

opass = before
opass = transform.SimplifyExpr()(opass)
opass = transform.EliminateCommonSubexpr()(opass)
opass_mem = simu_mem_from_relay(opass)
print(opass)

print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)