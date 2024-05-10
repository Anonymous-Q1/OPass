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

from Autotuning.gen import IncreGraphGenerator, IncreGenStatus
from Autotuning.pattern import PatternLearner, PatternCorpus
from Autotuning.util import viz2file
from GenCoG.gencog.spec import OpRegistry
from GenCoG.gencog.graph import print_relay

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=1, help='Random seed of graph generator.')  # 58 exists bug
    p.add_argument('-o', '--output', type=str, default='./out', help='Output directory.')
    p.add_argument('-c', '--corpus', type=str, default='./corpus', help='Pattern corpus directory.')
    p.add_argument('-n', '--num', type=int, default=100000, help='Number of generated files.')
    p.add_argument('-d', '--detail', action='store_true', help='Save runtime details.')
    args = p.parse_args()

def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    gen_status = IncreGenStatus()
    gen = IncreGraphGenerator(OpRegistry.ops(), rng)
    corpus = PatternCorpus(args.corpus, rng)
    learner = PatternLearner(rng, corpus)
    path = os.path.join(args.output, strftime('run-%Y%m%d-%H%M%S'))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')
    if not os.path.exists(path):
        os.mkdir(path)

    # Generation loop
    progress = tqdm(file=stdout)
    series_id = 0
    in_series_id = 0
    while progress.n < args.num:
        # Generate graph
        graph = gen.generate(gen_status)
        if graph is None:
            series_id += 1
            in_series_id = 0
            gen_status = IncreGenStatus()
            continue

        code = print_relay(graph)

        # Write code to case directory
        # case_id = str(progress.n)
        case_id = f'{series_id}-{in_series_id}'
        case_path = os.path.join(path, case_id)
        os.mkdir(case_path)
        codePath = os.path.join(case_path, 'code.txt')
        with open(codePath, 'w') as f:
            f.write(code)
        
        try:
            # Visualize the code
            viz2file(codePath)

            ### test
            # subg = learner.constr_subg([gen_status.last_opr])
            # with open(os.path.join(case_path, 'last_opt.txt'), 'w') as f:
            #     f.write(print_relay(subg))

            # Evaluate the coverage of new genetated graph
            cmd = ['python', 'opt_every_single_pass.py', f'-i={codePath}', f'-o={case_path}', f'-s={rng.integers(2 ** 63)}']
            # if args.detail:
            cmd.append('-d')
            run(cmd, env=env, check=True, timeout=600, stderr=open(os.devnull, 'w'), stdout=open(os.devnull, 'w'))

            # case_path = '/home/nie/RelayOpt/out/run-20230525-001953/0-0'
            cov_path = os.path.join(case_path, 'relay_opt_cov.json')
            with open(cov_path, 'r') as f:
                cov_stat = json.load(f)
            
            triggered_ps, upcov_ps = learner.detect_triggered_pass(case_path, cov_stat)
            print(triggered_ps)
            patterns = learner.detect_pattern(case_path, gen_status.last_opr, triggered_ps)
            upcov_ps += triggered_ps

            in_series_id += 1
        except:
            in_series_id += 1

        progress.update()

if __name__ == '__main__':
    parse_args()
    main()

'''
### test
try:
    with open(codePath, 'r') as f:
        mod = relay.parse(f.read())
    
except:
    print('##### Start Debug #####')
    print('---Oprs---')
    for opr in gen_status.oprs:
        print(opr.op_)
        print(opr)
        print(opr.inputs_)
        print(opr.outputs_)
    print('---Inputs---')
    for i in gen_status.inputs:
        print('-')
        print(i)
        print(i.value_)
        print(i.value_.def_)
    print('---Outputs---')
    for o in gen_status.outputs:
        print('-')
        print(o)
        print(o.value_)
    print('---Values---')
    for v in gen_status.value_lu.values:
        print('-')
        print(v)
        print(v.def_)
        print(v.uses_)
    
    raise Exception
'''