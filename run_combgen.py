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

from Autotuning.gen import CombGraphGenerator
from Autotuning.pattern import PatternCorpus
from Autotuning.util import viz2file
from GenCoG_cl.gencog.graph import print_relay

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=50, help='Random seed of graph generator.') 
    p.add_argument('-o', '--output', type=str, default='./out', help='Output directory.')
    p.add_argument('-c', '--corpus', type=str, default='./corpus', help='Pattern corpus directory.')
    p.add_argument('-n', '--num', type=int, default=100, help='Number of generated files.')
    args = p.parse_args()

def main():
    rng = Generator(PCG64(seed=args.seed))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    forbidden_ids = [51, 25, 52, 112, 46, 44, 15, 20, 50, 110, 36, 123, 0]
    corpus = PatternCorpus(args.corpus)
    corpus.patterns_ = {k: corpus.patterns_[k] for k in corpus.patterns_.keys() if k not in forbidden_ids}
    
    gen = CombGraphGenerator(corpus, rng)
    path = os.path.join(args.output, strftime('combine-%Y%m%d-%H%M%S'))
    if not os.path.exists(path):
        os.mkdir(path)

    progress = tqdm(file=stdout)
    while progress.n < args.num:
        # try:
        gmod = gen.generate(max_pattern_num=10)
        code = print_relay(gmod)
        
        # save the code file
        case_id = progress.n
        case_path = os.path.join(path, str(case_id))
        os.mkdir(case_path)
        code_path = os.path.join(case_path, 'code.txt')
        with open(code_path, 'w') as f:
            f.write(code)
        viz2file(code_path)

        # Evaluate whether this case is valid
        cmd = ['python3', '_run_combgen_ps.py', f'-p={code_path}', f'-s={rng.integers(2 ** 63)}']
        try:
            run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
        except:
            print('Check failed.')
            print(' '.join(cmd))
            os.system(f'rm -rf {case_path}')

        progress.update()

    # failed_patterns = []
    # for pattern in tqdm(corpus.patterns_.values()):
    #     if pattern.rule_ is None:
    #         continue
    #     code_path = pattern.path_
    #     cmd = ['python3', '_run_combgen_ps.py', f'-p={code_path}', f'-s={rng.integers(2 ** 63)}']
    #     try:
    #         run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
    #     except:
    #         print('Check failed.')
    #         print(' '.join(cmd))
    #         failed_patterns.append(pattern.idx_)
    # print(failed_patterns)
        
# cannot fit pattern: 15, 17, 24, 28

if __name__ == '__main__':
    parse_args()
    main()