from numpy.random import Generator, PCG64
from argparse import Namespace, ArgumentParser
from tvm.relay import parse, transform

from GenCoG_cl.gencog.graph import build_graph
from Autotuning.sequence import RandomRelaySeq
from Autotuning.util import simu_mem_footprint

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-p', '--path', type=str, help='Code path.')
    p.add_argument('-s', '--seed', type=int, default=52, help='Random seed of graph generator.')
    p.add_argument('-l', '--len', type=int, default=10, help='Length of generated sequences.')
    args = p.parse_args()

def main():
    rng = Generator(PCG64(seed=args.seed))
    seq_generator = RandomRelaySeq(rng)

    with open(args.path, 'r') as f:
        mod = parse(f.read())

    mod = transform.InferType()(mod)
    relay_seq = seq_generator.generate(max_len=args.len)
    mod = relay_seq.seq(mod)
    mod = transform.DynamicToStatic()(mod)

    mem = simu_mem_footprint(build_graph(mod)['main'])
    if not isinstance(mem, float):
        raise Exception('Dynamic memory allocation.')
    
    print(f'{mem} mb')

if __name__ == '__main__':
    parse_args()
    main()