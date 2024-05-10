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
x = relay.var("x", shape=[1, 3, 224, 224], dtype="int8")
w = relay.var("w", shape=[16, 3, 5, 5], dtype="int8")
one = relay.const(1.0)
zero = relay.const(0)

x = relay.qnn.op.dequantize(x, relay.const(2.0), zero)
w = relay.qnn.op.dequantize(w, relay.const(0.5), zero)
x = relay.transpose(x, axes=[0, 1, 2, 3])
op = relay.op.nn.conv2d(
    x,
    w,
    kernel_size=[5, 5],
)
op = relay.layout_transform(op, "NCHW", "NCHW")
op = relay.qnn.op.quantize(op, one, zero, out_dtype='int8')
before = tvm.IRModule.from_expr(op)
before = transform.InferType()(before)
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
case_dir = os.path.join(root, '7')
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
# default = transform.SimplifyInference()(default)
default = transform.SimplifyExpr()(default)
default = transform.FuseOps(4)(default)
# default = transform.ToMixedPrecision()(default)
default_mem = simu_mem_from_relay(default)
# print(default)

opass = before
opass = transform.SimplifyExpr()(opass)
opass = transform.FakeQuantizationToInteger()(opass)
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