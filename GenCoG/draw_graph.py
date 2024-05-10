from gencog.graph.relay import build_graph
from gencog.graph.viz import visualize

from tvm import parser
from numpy.random import Generator, PCG64
from typing import Dict, Optional, List
from tvm import relay, tir, parser
import os
from argparse import ArgumentParser

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)


def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

argparser = ArgumentParser()
argparser.add_argument('-f', '--file', type=str, help='File Path.')
args = argparser.parse_args()
with open(args.file) as f:
    mod = parser.parse(f.read())

rng = Generator(PCG64(seed=42))

# Generate input parameters
main_fn = mod['main']
inputs = gen_tensor_value_dict(main_fn.params[0:1], rng)
params = gen_tensor_value_dict(main_fn.params[1:], rng)

graph = build_graph(mod, params)
visualize(graph, 'graph', os.path.dirname(args.file))
