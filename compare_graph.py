from numpy.random import Generator, PCG64
from argparse import ArgumentParser

import tvm
from tvm import parser

from Autotuning.graph import GraphAbstractor, GraphComparer

# Parse arguments
p = ArgumentParser()
p.add_argument('-a', '--fileA', type=str, default='/home/nie/RelayOpt/GenCoG/out/run-20230509-014533/1/code.txt', help='Path of input file A.')
p.add_argument('-b', '--fileB', type=str, default='/home/nie/RelayOpt/GenCoG/out/run-20230509-014533/1/code_1.txt', help='Path of input file B.')
p.add_argument('-s', '--seed', type=int, default=58, help='Random seed of graph generator.')
args = p.parse_args()

rng = Generator(PCG64(seed=args.seed))

filepath1 = args.fileA
filepath2 = args.fileB
with open(filepath1, 'r') as f:
    mod1 = parser.parse(f.read())
with open(filepath2, 'r') as f:
    mod2 = parser.parse(f.read())
abs1 = GraphAbstractor('graph').get_graph_from_mod(mod1, rng)
abs2 = GraphAbstractor('graph').get_graph_from_mod(mod2, rng)

comparer = GraphComparer(abs1, abs2)
# if not comparer.compare():
#     exit(1)
print(comparer.compare())
# print(comparer.compres)