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
def gen_intermediate_tuple(x):
    y1 = relay.add(x, relay.const(1, "float32"))
    y2 = relay.add(x, relay.const(1, "float32"))
    # y3 = relay.add(x, relay.const(1, "float32"))
    concat = relay.concatenate((y1, y2), axis=1)
    out = relay.add(concat, relay.const(1, "float32"))
    return out

def gen_consecutive_tuple(x):
    y1 = gen_intermediate_tuple(x)
    y2 = gen_intermediate_tuple(x)
    # y3 = gen_intermediate_tuple(x)
    concat = relay.concatenate((y1, y2), axis=1)
    return concat

def before(x):
    concat = gen_consecutive_tuple(x)
    pooled = relay.nn.max_pool2d(concat, pool_size=(2, 2), strides=(2, 2), padding=(0, 0))
    out = relay.add(pooled, relay.const(1, "float32"))
    out2 = relay.add(out, relay.const(1, "float32"))
    out_tup = relay.Tuple((out, out2))
    return relay.Function(relay.analysis.free_vars(out_tup), out_tup)

dshape = (1, 16, 64, 64)
x = relay.var("x", shape=dshape)
orig = before(x)
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
case_dir = os.path.join(root, '3_1')
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
# print(opass)

ref = transform.FuseOps(4)(before)
ref = transform.ToMixedPrecision()(ref)
ref_mem = simu_mem_from_relay(ref)

print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)
print('Ref:', ref_mem, 'mem;', (origin_mem-ref_mem)/origin_mem)

# tmp_path = os.path.join(case_dir, 'tmp.txt')
# with open(tmp_path, 'w') as f:
#     f.write(default.astext())
# viz2file(tmp_path)