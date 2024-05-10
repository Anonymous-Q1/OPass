import os
from subprocess import run

tvm_source_path = '/home/nie/tvm-gcov/tvm/'

# Run the script
root = './'
script_path = '/home/nie/RelayOpt/_test/opt.py'

env = os.environ.copy()
env['PYTHONPATH'] = os.path.join(root, 'python')
env['CONDA_PREFIX'] = '/home/nie/miniconda3'
env['CONDA_PROMPT_MODIFIER'] = '(base)'
env['CONDA_DEFAULT_ENV'] = 'base'
env['CONDA_SHLVL'] = '3'
env['PATH'] = '/home/nie/miniconda3/bin:/home/nie/miniconda3/condabin:/home/nie/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin'

cmd_env = f'export TVM_HOME={tvm_source_path} && export PYTHONPATH=$TVM_HOME/python && '
cmd = cmd_env + f'python {script_path}'
run(cmd, env=env, shell=True, check=True, timeout=60, stderr=open(os.devnull, 'w'))

# Collect the coverage
cov_out_dir = '/home/nie/RelayOpt/GenCoG/out/run-20230509-014533/2/'

tvm_env = os.environ.copy()
cov_collect_cmd = ['gcovr', '-r', tvm_source_path, '--json-summary-pretty', '-o', cov_out_dir, '--gcov-ignore-parse-errors', '-d']
run(cov_collect_cmd, env=tvm_env, check=True, timeout=600, stderr=open(os.devnull, 'w'))

# Read the coverage of opt pass code
import json
cov_out_dir = '/home/nie/RelayOpt/GenCoG/out/run-20230509-014533/2/'

cov_stats = {}
cov_file = os.path.join(cov_out_dir, 'summary_coverage.json')
with open(cov_file, 'r') as f:
    cov = json.load(f)
for file in cov['files']:
    line_cov = file['line_covered']
    filename = file['filename']
    if filename.startswith('src/relay/transforms/') and line_cov != 0:
        cov_stats[filename[21:]] = line_cov
print(cov_stats)
