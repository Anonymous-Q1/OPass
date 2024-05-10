import os
from argparse import ArgumentParser
from numpy.random import Generator, PCG64
from subprocess import run
from Autotuning.sequence import RelayPassTable

# Parse arguments
p = ArgumentParser()
p.add_argument('-i', '--input', type=str, default='/home/nie/RelayOpt/out/run-20230509-014533/2/code.txt', help='Input file path.')
p.add_argument('-o', '--output', type=str, default='/home/nie/RelayOpt/out/run-20230509-014533/2/', help='Directory of generated coverage file.')
p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
p.add_argument('-c', '--src', type=str, default='/home/nie/tvm-gcov/tvm/', help='Directory of TVM source code.')
p.add_argument('-s', '--seed', type=int, default=58, help='Random seed of graph generator.')
p.add_argument('-d', '--detail', action='store_true', help='Save runtime details.')
args = p.parse_args()

rng = Generator(PCG64(seed=args.seed))

env = os.environ.copy()
env['PYTHONPATH'] = os.path.join(args.root, 'python')
env['CONDA_PREFIX'] = '/home/nie/miniconda3'
env['CONDA_PROMPT_MODIFIER'] = '(base)'
env['CONDA_DEFAULT_ENV'] = 'base'
env['CONDA_SHLVL'] = '3'
env['PATH'] = '/home/nie/miniconda3/bin:/home/nie/miniconda3/condabin:/home/nie/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'

cmd_env = f'export TVM_HOME={args.src} && export PYTHONPATH=$TVM_HOME/python && '
for pass_name in RelayPassTable.NameTable:
    try:
        cmd = cmd_env + f'python opt_single_pass.py -i={args.input} -s={rng.integers(2 ** 63)} -p={pass_name}'
        if args.detail:
            outpath = os.path.join(os.path.dirname(args.input), pass_name+'.txt')
            cmd += f' -o={outpath}'
        run(cmd, env=env, shell=True, check=True, timeout=60, stderr=open(os.devnull, 'w'))
    except:
        # print(cmd)
        print(f'Run pass {pass_name} failed.')

# Collect the coverage
print('Collecting coverage...')
tvm_env = os.environ.copy()
cov_file = os.path.join(args.output, 'summary_coverage.json')
cov_collect_cmd = ['gcovr', '-r', args.src, '--json-summary-pretty', '-o', cov_file, '--gcov-ignore-parse-errors']  # , '-d'
run(cov_collect_cmd, env=tvm_env, check=True, timeout=600, stderr=open(os.devnull, 'w'))

# Read the coverage of opt pass code
import json
cov_stats = {}
with open(cov_file, 'r') as f:
    cov = json.load(f)
for file in cov['files']:
    line_cov = file['line_covered']
    bran_cov = file['branch_covered']
    filename = str(file['filename'])
    if filename.startswith('src/relay/transforms/') and filename[21:] in RelayPassTable.SrcTable:    #  and line_cov != 0
        cov_stats[filename[21:]] = [line_cov, bran_cov]
    elif filename == 'src/relay/parser/parser.cc':
        cov_stats['parser.cc'] = [line_cov, bran_cov]
    elif filename == 'src/relay/collage/collage_partitioner.cc':
        cov_stats['collage_partitioner.cc'] = [line_cov, bran_cov]
    elif filename == 'src/relay/backend/vm/lambda_lift.cc':
        cov_stats['lambda_lift.cc'] = [line_cov, bran_cov]
    elif filename == 'src/relay/backend/vm/removed_unused_funcs.cc':
        cov_stats['removed_unused_funcs.cc'] = [line_cov, bran_cov]

with open(os.path.join(args.output, 'relay_opt_cov.json'), 'w') as f:
    json.dump(cov_stats, f, indent='')