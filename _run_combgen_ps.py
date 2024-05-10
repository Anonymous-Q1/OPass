'''
Combine patterns to graph, aiming to generate benchmarks suitable for opt seq customization.

跑之前记得把config里的max_dim调成512!!!
'''

import os
import json
from time import strftime
from tqdm import tqdm
from sys import stdout
from subprocess import run
from collections import defaultdict
from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64
from tvm import relay
import tvm

from Autotuning.gen import CombGraphGenerator
from Autotuning.pattern import PatternCorpus
from Autotuning.util import viz2file, gen_tensor_value, gen_tensor_value_dict
from GenCoG_cl.gencog.graph import print_relay

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=46, help='Random seed of graph generator.')  # 58 exists bug
    p.add_argument('-p', '--path', type=str, help='Code path.')
    # p.add_argument('-c', '--corpus', type=str, default='./corpus', help='Pattern corpus directory.')
    args = p.parse_args()

def main():
    rng = Generator(PCG64(seed=args.seed))
    # corpus = PatternCorpus(args.corpus)
    # gen = CombGraphGenerator(corpus, rng)
    
    # gmod = gen.generate(max_pattern_num=40)
    # code = print_relay(gmod)

    # check the generated code
    with open(args.path, 'r') as f:
        code = f.read()
    try:
        mod = relay.parse(code)
        mod = relay.transform.InferType()(mod)
        mod = relay.transform.DynamicToStatic()(mod)
        main_fn = mod['main']
        params = gen_tensor_value_dict(main_fn.params[1:], rng)
        with tvm.transform.PassContext(opt_level=4) as PC:
            _ = relay.build(mod, target='llvm', params=params)
    except Exception as e:
        print(e)
        exit(1)
        
    # # save the code file
    # case_path = args.output
    # os.mkdir(case_path)
    # code_path = os.path.join(case_path, 'code.txt')
    # with open(code_path, 'w') as f:
    #     f.write(code)
    # viz2file(code_path)

if __name__ == '__main__':
    parse_args()
    main()