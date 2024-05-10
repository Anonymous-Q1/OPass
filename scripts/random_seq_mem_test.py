'''
This file generate a lot random optimization sequence for codes and observe the memory footprint after optimization.
'''

import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError, check_output
import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm
import json
import psutil

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-d', '--directory', type=str, default='/home/nie/RelayOpt/out/combine-20230626-215024/', help='Case directory.')
    p.add_argument('-s', '--seed', type=int, default=52, help='Random seed of graph generator.')
    p.add_argument('-l', '--len', type=int, default=10, help='Length of generated sequences.')
    p.add_argument('-n', '--num', type=int, default=50, help='# of generated sequences for each case.')
    args = p.parse_args()

def resolve_script_res(r):
    r = str(r, 'utf-8').strip()
    assert r.endswith(' mb')
    exec_mem = float(r[:-3].split('/')[0])
    simu_mem = float(r[:-3].split('/')[1])
    return exec_mem, simu_mem

def main():
    rng = Generator(PCG64(seed=args.seed))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    for case_dir in os.listdir(args.directory):
        case_path = os.path.join(args.directory, case_dir)
        print('##########################')
        print('Testing', case_path)
        
        # Try to read the test result and analyze
        if os.path.exists(os.path.join(case_path, 'TESTED')):
            # os.system('rm -rf ' + os.path.join(case_path, 'TESTED'))
            if os.path.exists(os.path.join(case_path, 'mem_test.json')):
                with open(os.path.join(case_path, 'mem_test.json'), 'r') as f:
                    tes_res = json.load(f)
                if tes_res['random_best_simu']['sim_mem'] < tes_res['Unoptimized']['sim_mem'] and \
                    tes_res['random_best_simu']['sim_mem'] < tes_res['DefaultOpt']['sim_mem']:
                    print('Unoptimized: %s/%s mb.'%(str(tes_res['Unoptimized']['mem']), str(tes_res['Unoptimized']['sim_mem'])))
                    print('DefaultOpt: %s/%s mb.'%(str(tes_res['DefaultOpt']['mem']), str(tes_res['DefaultOpt']['sim_mem'])))
                    print('Best Random Exec: %s/%s mb.'%(str(tes_res['random_best_exec']['mem']), str(tes_res['random_best_exec']['sim_mem'])))
                    print('Seq:', tes_res['random_best_exec']['seq'])
                    print('Best Random Simu: %s/%s mb.'%(str(tes_res['random_best_simu']['mem']), str(tes_res['random_best_simu']['sim_mem'])))
                    print('Seq:', tes_res['random_best_simu']['seq'])
            continue

        # Mark this case as tested
        with open(os.path.join(case_path, 'TESTED'), 'w') as _:
            pass
        # Test and Record
        tes_res = {}
        try:
            cmd = ['python3', './_run_random_ps.py', f'-d={case_path}', f'-o=0', f'-s={rng.integers(2 ** 63)}', '-M']
            r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
            exec_mem_ref, simu_mem_ref = resolve_script_res(r)
            print(f'Unoptimized: {exec_mem_ref}/{simu_mem_ref} mb.')
            tes_res['Unoptimized'] = {'mem': exec_mem_ref, 'sim_mem':simu_mem_ref}

            cmd = ['python3', './_run_random_ps.py', f'-d={case_path}', f'-o=4', f'-s={rng.integers(2 ** 63)}', '-M']
            r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
            exec_mem, simu_mem = resolve_script_res(r)
            print(f'DefaultOpt: {exec_mem}/{simu_mem} mb.')
            tes_res['DefaultOpt'] = {'mem': exec_mem, 'sim_mem':simu_mem}
            
            min_exec_mem = 10000000000
            min_simu_mem = 10000000000
            for i in tqdm(range(args.num)):
                try:
                    cmd = ['python3', './_run_random_ps.py', f'-d={case_path}', f'-l={args.len}', f'-s={rng.integers(2 ** 63)}', '-M']
                    r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
                    exec_mem, simu_mem = resolve_script_res(r)
                    
                    with open(os.path.join(case_path, 'seq.txt'), 'r') as f:
                        tes_res[str(i)] = {'mem': exec_mem, 'sim_mem':simu_mem, 'seq': eval(f.read())}

                    if exec_mem < min_exec_mem:
                        min_exec_mem = exec_mem
                        tes_res['random_best_exec'] = tes_res[str(i)]

                    if simu_mem < min_simu_mem:
                        min_simu_mem = simu_mem
                        tes_res['random_best_simu'] = tes_res[str(i)]
                except:
                    continue
            
            with open(os.path.join(case_path, 'mem_test.json'), 'w') as f:
                json.dump(tes_res, f, indent='')

            print('Best Random Exec: %s/%s mb.'%(str(tes_res['random_best_exec']['mem']), str(tes_res['random_best_exec']['sim_mem'])))
            print('Seq:', tes_res['random_best_exec']['seq'])
            print('Best Random Simu: %s/%s mb.'%(str(tes_res['random_best_simu']['mem']), str(tes_res['random_best_simu']['sim_mem'])))
            print('Seq:', tes_res['random_best_simu']['seq'])
            
        except Exception as e:
            print(e)
            print('Error:', ' '.join(cmd))
            continue

if __name__ == '__main__':
    parse_args()
    main()
