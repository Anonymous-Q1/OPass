import sys
from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64

import tvm
from tvm import relay

from Autotuning.sequence import RelaySeq, RelayPassSelector

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-s', '--seed', type=int, default=58, help='Random seed of graph generator.')
    p.add_argument('-i', '--input', type=str, help='Input file path.')
    p.add_argument('-o', '--output', type=str, default='', help='Output file path.')
    p.add_argument('-p', '--passname', type=str, help='Pass name.')
    args = p.parse_args()

def main():
    rng = Generator(PCG64(seed=args.seed))

    with open(args.input, 'r') as f:
        mod = relay.parse(f.read())

    passSelector = RelayPassSelector(rng)
    try:
        p = passSelector.wrap_pass(args.passname)
    except Exception as e:
        print(f'Cannot generate a Relay OPT pass for the name {args.passname}:')
        print(e)
        exit(1)
    
    relaySeq = RelaySeq()
    if args.passname in ['FuseOps', 'CanonicalizeCast']:
        relaySeq.append(passSelector.wrap_pass('SimplifyInference'))
    relaySeq.append(p)

    try:
        with tvm.transform.PassContext(opt_level=5):
            mod = relaySeq.seq(mod)
    except:
        print(f'Error occurs when apply OPT pass {args.passname}.')
        exit(1)
    
    if args.output != '':
        with open(args.output, 'w') as f:
            f.write(mod.astext())

if __name__ == '__main__':
    parse_args()
    main()
    sys.exit()
