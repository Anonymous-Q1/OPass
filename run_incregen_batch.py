'''
Incrementally generate graph and analyze code coverage to learn patterns.
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
from typing import Dict

from Autotuning.gen import IncreGraphGenerator, IncreGenStatus
from Autotuning.pattern import PatternLearner, PatternCorpus
from Autotuning.util import viz2file
from GenCoG.gencog.spec import OpRegistry
from GenCoG.gencog.graph import print_relay

args = Namespace()
tvm_root = '/home/nie/tvm-gcov/tvm/'
gcda_save_dir = '/home/nie/tvm-gcov/backup/tvm-gcda/'
cov_backup_path = '/home/nie/tvm-gcov/backup/max_cov.json'

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=49, help='Random seed of graph generator.')  # 58 exists bug
    p.add_argument('-o', '--output', type=str, default='./out', help='Output directory.')
    p.add_argument('-c', '--corpus', type=str, default='./corpus', help='Pattern corpus directory.')
    p.add_argument('-n', '--num', type=int, default=100000, help='Number of generated files.')
    p.add_argument('-b', '--batch', type=int, default=20, help='Batch size.')
    # p.add_argument('-d', '--detail', action='store_true', help='Save runtime details.')
    args = p.parse_args()

def backup_gcda(backup_dir:str, source_dir:str):
    if os.path.exists(backup_dir):
        os.system('rm -rf ' + backup_dir)
    os.mkdir(backup_dir)

    search_list = [source_dir]
    while search_list:
        current_dir = search_list.pop(0)
        for name in os.listdir(current_dir):
            path = os.path.join(current_dir, name)
            if os.path.isdir(path):
                os.mkdir(backup_dir + path[len(source_dir):])
                search_list.append(path)
            elif path.endswith('.gcda'):
                os.system(f'cp {path} {backup_dir + path[len(source_dir):]}')

def from_bk_gcda(backup_dir:str, source_dir:str):
    if not os.path.exists(backup_dir):
        raise Exception('No gcda backup.')
    
    search_list = [backup_dir]
    while search_list:
        current_dir = search_list.pop(0)
        for name in os.listdir(current_dir):
            path = os.path.join(current_dir, name)
            if os.path.isdir(path):
                search_list.append(path)
            elif path.endswith('.gcda'):
                os.system(f'cp {path} {source_dir + path[len(backup_dir):]}')

def backup_cov_stat(backup_path:str, cov_stat:Dict[str, int]):
    with open(backup_path, 'w') as f:
        json.dump(cov_stat, f)

def from_bk_cov_stat(backup_path:str) -> Dict[str, int]:
    cov_stat = defaultdict(int)
    with open(backup_path, 'r') as f:
        cov = json.load(f)
    for k in cov:
        cov_stat[k] = cov[k]
    return cov_stat

def get_cov_stat(codePath:str, case_path:str, rng:Generator, env:Dict):
    cmd = ['python', 'opt_every_single_pass.py', f'-i={codePath}', f'-o={case_path}', f'-c={tvm_root}', f'-s={rng.integers(2 ** 63)}', '-d']
    run(cmd, env=env, check=True, timeout=600, stderr=open(os.devnull, 'w'), stdout=open(os.devnull, 'w'))
    cov_path = os.path.join(case_path, 'relay_opt_cov.json')
    with open(cov_path, 'r') as f:
        cov_stat = json.load(f)
    return cov_stat

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
        # backup the gcda files
        backup_gcda(gcda_save_dir, tvm_root)
        backup_cov_stat(cov_backup_path, learner.max_cov)

        # Generate graph with #args.batch steps incrementation.
        for _ in range(args.batch):
            graph = gen.generate(gen_status)
            if graph is None:
                break
        if graph is None:
            series_id += 1
            in_series_id = 0
            gen_status = IncreGenStatus()
            continue

        # Write code to case directory
        code = print_relay(graph)
        case_id = f'{series_id}-{in_series_id+args.batch-1}'
        case_path = os.path.join(path, case_id)
        os.mkdir(case_path)
        codePath = os.path.join(case_path, 'code.txt')
        with open(codePath, 'w') as f:
            f.write(code)
        
        # Evaluate the coverage of new genetated graph, and check if updated.
        cov_stat = get_cov_stat(codePath, case_path, rng, env)
        triggered_ps, _ = learner.detect_triggered_pass(case_path, cov_stat)
        # os.system(f'rm -rf {case_path}')

        # If some passes are triggered in this batch, then check when it happened.
        if triggered_ps:
            print('#################################')
            print('Detected triggered pass:', triggered_ps)

            # Start to backtrack
            from_bk_gcda(gcda_save_dir, tvm_root)
            learner.max_cov = from_bk_cov_stat(cov_backup_path)
            print('Start backtracking...')

            for incre_num in range(1, args.batch+1):
                print(f'Backtracking {incre_num}.')
                recent_graph = learner.constr_subg(gen_status.oprs[:len(gen_status.oprs) -(args.batch - incre_num)])
                recent_last_opt = recent_graph.oprs_[-1]

                # Write cide to case directory
                recent_code = print_relay(recent_graph)
                recent_case_id = f'{series_id}-{in_series_id+incre_num-1}'
                recent_case_path = os.path.join(path, recent_case_id)
                if not os.path.exists(recent_case_path):
                    os.mkdir(recent_case_path)
                recent_codePath = os.path.join(recent_case_path, 'code.txt')
                with open(recent_codePath, 'w') as f:
                    f.write(recent_code)
                
                # Visualize the code
                viz2file(recent_codePath)

                # Evaluate the coverage of backtracked graph, and try to learn patterns if updated.
                recent_cov_stat = get_cov_stat(recent_codePath, recent_case_path, rng, env)
                recent_triggered_ps, _ = learner.detect_triggered_pass(recent_case_path, recent_cov_stat)
                patterns = learner.detect_pattern(recent_case_path, recent_last_opt, recent_triggered_ps)
                if len(patterns) > 0:
                    print(f'Learned {len(patterns)} patterns from case {recent_case_path}.')

        in_series_id += args.batch
        progress.update()

if __name__ == '__main__':
    parse_args()
    main()
