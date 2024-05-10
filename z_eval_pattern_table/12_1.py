# To construct a pattern table and corresponding seq impact
# Following static/1/
import os
import tvm
from tvm import relay
from tvm.relay import transform
from Autotuning.util import viz2file, simu_mem_from_relay

root = '/home/nie/RelayOpt/eval/pattern_table/'

'''
Construct pattern
'''
# b, i, j, k = 1, 100, 200, 300
# x = relay.var("x", shape=(b, i, k), dtype='float32')
# w1 = relay.var("w1", shape=(b, j, k), dtype='float32')
# w2 = relay.var("w2", shape=(b, j, k), dtype='float32')
# w3 = relay.var("w2", shape=(b, j, k), dtype='float32')
# y1 = relay.nn.batch_matmul(x, w1)
# y2 = relay.nn.batch_matmul(x, w2)
# y3 = relay.nn.batch_matmul(x, w3)
# # c = relay.ones((b, i, j), dtype='float32')
# y1 = relay.add(y1, y2)
# y = relay.Tuple([y1, y3])
# func = relay.Function([x, w1, w2, w3], y)

def before(x, w1, w2, scale1, scale2, bias):
    args = [x, w1, scale1]
    y1 = relay.nn.conv2d(x, w1)
    y1 = relay.multiply(y1, scale1)
    y1 = relay.nn.relu(y1)
    y2 = relay.nn.conv2d(x, w1)
    y2 = relay.multiply(y2, scale1)
    y2 = relay.nn.relu(y2)
    # y2 = relay.add(y2, bias)
    y3 = relay.nn.conv2d(x, w1)
    y3 = relay.multiply(y3, scale1)
    y3 = relay.nn.relu(y3)
    y = relay.Tuple((y1, y2, y3))
    return relay.Function(args, y)
x_shape, channels1, channels2 = (1, 4, 16, 16), 4, 8
x = relay.var("x", shape=x_shape)
in_c = x_shape[1]
w1 = relay.var("w1", shape=(channels1, in_c, 1, 1))
w2 = relay.var("w2", shape=(channels2, in_c, 1, 1))
scale1 = relay.var("scale1", shape=(channels1, 1, 1))
scale2 = relay.var("scale2", shape=(channels2, 1, 1))
bias = relay.var("bias", shape=(channels2, 1, 1))
y_before = before(x, w1, w2, scale1, scale2, bias)

before = tvm.IRModule.from_expr(y_before)
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
case_dir = os.path.join(root, '12_1')
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
default = transform.CombineParallelConv2D()(before)
# default = transform.SimplifyExpr()(default)
default = transform.FuseOps()(default)
# default = transform.ToMixedPrecision()(default)
default_mem = simu_mem_from_relay(default)
# print(default)

# opass = transform.SimplifyExpr()(before)
opass = transform.FuseOps()(before)
# opass = transform.CombineParallelBatchMatmul()(opass)
# opass = transform.ToMixedPrecision()(opass)
opass_mem = simu_mem_from_relay(opass)
print(opass)

print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)

# tmp_path = os.path.join(case_dir, 'tmp.txt')
# with open(tmp_path, 'w') as f:
#     f.write(default.astext())
# viz2file(tmp_path)