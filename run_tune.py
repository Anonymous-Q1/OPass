import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError, check_output
import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm
from Autotuning.tune import Tuner
from Autotuning.pattern import PatternCorpus

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-d', '--directory', type=str, default='/home/nie/RelayOpt/out/combine-20230626-232349/16', help='Case directory.')
    p.add_argument('-s', '--seed', type=int, default=52, help='Random seed of graph generator.')
    p.add_argument('-c', '--corpus', type=str, default='./corpus', help='Pattern corpus directory.')
    args = p.parse_args()

def main():
    rng = Generator(PCG64(seed=args.seed))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')
    corpus = PatternCorpus(args.corpus, rng)
    tuner = Tuner(corpus, rng)
    seq = tuner.tune(args.directory, max_step=100)
    seq_path = os.path.join(args.directory, 'seq.txt')
    print(seq.info)
    seq.save(seq_path)
    cmd = ['python3', './_run_random_ps.py', f'-d={args.directory}', f'-q={seq_path}', f'-s={rng.integers(2 ** 63)}', '-M']
    print(' '.join(cmd))

if __name__ == '__main__':
    parse_args()
    main()
