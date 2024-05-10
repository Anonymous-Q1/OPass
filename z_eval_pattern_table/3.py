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
def conv(data):
    y = relay.nn.conv2d(data, relay.var("w"), kernel_size=(3, 3), padding=(1, 1), channels=16)
    return relay.nn.relu(data=y)

def inception_like(data):
    c0 = conv(data)
    c1 = conv(data)
    return relay.concatenate((c0, c1), axis=1)

def before_func(dshape):
    x = relay.var("x", shape=dshape)
    in1 = inception_like(x)
    in2 = inception_like(in1)
    return relay.Function(relay.analysis.free_vars(in2), in2)

orig = before_func([1, 3, 64, 64])
before = tvm.IRModule.from_expr(orig)
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
case_dir = os.path.join(root, '3')
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
default = transform.FuseOps(4)(before)
# default = transform.SimplifyExpr()(default)
# default = transform.ToMixedPrecision()(default)
# default = transform.ToMixedPrecision()(default)
default_mem = simu_mem_from_relay(default)
# print(default)

# opass = transform.SimplifyExpr()(before)
opass = transform.ToMixedPrecision()(before)
opass = transform.FuseOps(4)(opass)
# opass = transform.ToMixedPrecision()(opass)
opass_mem = simu_mem_from_relay(opass)
print(opass)

print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)

# tmp_path = os.path.join(case_dir, 'tmp.txt')
# with open(tmp_path, 'w') as f:
#     f.write(default.astext())
# viz2file(tmp_path)