from numpy.random import Generator, PCG64
from typing import List
from argparse import Namespace, ArgumentParser
from tvm.relay import parse, transform, Var, build
import tvm
import json
import os

from Autotuning.tune.transfer_eval import TranferGraph

from Autotuning.util import cal_tvm_mem, simu_mem_from_relay, serenity_mem_from_relay

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-p', '--path', type=str, default='/home/nie/RelayOpt/out/static/combine-20231006-231536-8/code.txt', help='Code path.')
    p.add_argument('-s', '--seed', type=int, default=55, help='Random seed of graph generator.')
    p.add_argument('-e', '--epochs', type=int, default=100, help='Total iteration number.')
    args = p.parse_args()

def gen_tensor_value(var: Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)

def gen_tensor_value_dict(params: List[Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

def main():
    rng = Generator(PCG64(seed=args.seed))

    with open(args.path, 'r') as f:
        mod = parse(f.read())
    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)
    with open(args.path+'.txt', 'w') as f:
        f.write(mod.astext())

    profiler = simu_mem_from_relay

    # print(profiler(mod), 'MB')
    # print(simu_mem_from_relay(mod), 'MB')
    # print(cal_tvm_mem(mod), 'MB')

    # compilation check
    # main_fn = mod['main']
    # params = gen_tensor_value_dict(main_fn.params[1:], rng)
    # with tvm.transform.PassContext(opt_level=4) as PC:
    #     _ = build(mod, target='llvm', params=params)

    transferG = TranferGraph(args.path+'.txt', rng, profiler=profiler) # , profiler=cal_tvm_mem
    mem, seq = transferG.run(args.epochs)

if __name__ == '__main__':
    parse_args()
    main()

'''
static/combine-20231006-231536-0
Inherent rate: 0.8637015781922525.
Newly generation rate: 0.14183891660727013

static/combine-20231006-231536-6
Inherent rate: 0.7364238410596027.
Newly generation rate: 0.3660205245153934
'''

# TODO: 1. eliminate propagation. 2. Simulated annealing 