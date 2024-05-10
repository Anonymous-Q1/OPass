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
t = relay.TensorType([], "float32")
d = relay.Var("d", t)
double = relay.Function([d], d + d)
orig = double(relay.const(4.0))

before = tvm.IRModule.from_expr(orig)
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
case_dir = os.path.join(root, '10')
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
# default = transform.ToGraphNormalForm()(default)
# default = transform.FoldConstant()(default)
# default = transform.ToMixedPrecision()(default)
# default = transform.FuseOps(4)(default)
default_mem = simu_mem_from_relay(default)
print(default)

opass = before
opass = transform.PartialEvaluate()(opass)
opass = transform.InferType()(opass)
opass = transform.DeadCodeElimination()(opass)
opass = transform.InferType()(opass)
# opass = transform.FuseOps(4)(opass)
opass_mem = simu_mem_from_relay(opass)
print(opass)
# 
print('Default:', default_mem, 'mem;', (origin_mem-default_mem)/origin_mem)
print('OPass:', opass_mem, 'mem;', (origin_mem-opass_mem)/origin_mem)

# tmp_path = os.path.join(case_dir, 'tmp.txt')
# with open(tmp_path, 'w') as f:
#     f.write(default.astext())
# viz2file(tmp_path)