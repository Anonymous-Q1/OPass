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
def before():
    x = relay.var("x", shape=(1, 16))

    x1 = relay.var("x1", shape=(1, 16))
    y1 = relay.nn.relu(x1)
    y1 = relay.add(y1, relay.const(1.0, "float32"))
    f1 = relay.Function([x1], y1)

    x2 = relay.var("x2", shape=(1, 16))
    y2 = relay.nn.relu(x2)
    y2 = relay.add(y2, relay.const(1.0, "float32"))
    f2 = relay.Function([x2], y2)

    z1 = relay.Call(f1, [x])
    z2 = relay.Call(f2, [x])
    z = relay.add(z1, z2)
    f = relay.Function([x], z)
    return f
func = before()
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
case_dir = os.path.join(root, '5_1')
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