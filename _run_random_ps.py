import os
from argparse import ArgumentParser

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm

from tvm import relay, tir
from tvm.ir.transform import Sequential

from Autotuning.debug import ModuleRunner, ModuleError
from Autotuning.sequence import RandomRelaySeq, RelaySeq
from Autotuning.util import viz2file

# Parse arguments
parser = ArgumentParser()
parser.add_argument('-d', '--directory', type=str, help='Case directory.')
parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed.')
parser.add_argument('-l', '--length', type=int, default=5, help='Length of sequence.')
parser.add_argument('-q', '--sequence', type=str, default='', help='File path of relay optimization sequence. When use this param, -l are not used')
parser.add_argument('-o', '--opt_level', type=int, default=-1, help='Default optimization level. When use this param, -q and -l are not used.')
parser.add_argument('-M', '--memory_profile', action='store_true', default=False, help='Profile the memory footprint.')
parser.add_argument('-T', '--time_eval', action='store_true', default=False, help='Evaluation the time cost.')
parser.add_argument('-S', '--save', action='store_true', default=False, help='Save the optimized code to the same dir named "opt_code.txt".')
parser.add_argument('-V', '--visualize', action='store_true', default=False, help='Visualize the code to the same dir named "code.gv.pdf".')
args = parser.parse_args()

# Initialize runner
rng = Generator(PCG64(seed=args.seed))
runner = ModuleRunner(rng)
seq_generator = RandomRelaySeq(rng)
opt_level=4

with open(os.path.join(args.directory, 'code.txt'), 'r') as f:
    code = f.read()

try:
    # Generate or load a sequence.
    relay_seq = None
    if args.opt_level != -1:
        relay_seq = None
        opt_level = args.opt_level
    elif args.sequence != '':
        with open(args.sequence, 'r') as f:
            seq_info = eval(f.read())
        relay_seq = RelaySeq()
        relay_seq.from_info(seq_info)
    else:
        relay_seq = seq_generator.generate(max_len=args.length)

    # Run and evaluate the sequence.
    runner.run(code, seq=relay_seq, opt_level=opt_level)
    if args.memory_profile:
        exec_mem, sim_mem = runner.eval_mem(code, seq=relay_seq, opt_level=opt_level)
        print(f'{exec_mem:.4f}/{sim_mem:.4f} mb')
    if args.time_eval:
        exec_time = runner.eval_time(code, seq=relay_seq, opt_level=opt_level)
        print(f'{exec_time:.4f} ms')
    if args.save:
        runner.opt(code, seq=relay_seq, opt_level=opt_level, save_path=os.path.join(args.directory, 'opt_code.txt'))
        if args.visualize:
            viz2file(os.path.join(args.directory, 'opt_code.txt'), args.seed)
    if args.visualize:
        viz2file(os.path.join(args.directory, 'code.txt'), args.seed)

    if relay_seq != None:
        relay_seq.save(os.path.join(args.directory, 'seq.txt'))

# Error handling
except ModuleError as err:
    bug_dir = os.path.join(args.directory, 'bugs')
    if not os.path.exists(bug_dir):
        os.mkdir(bug_dir)
    err.report(os.path.join(bug_dir, str(len(os.listdir(bug_dir)))))
    exit(1)
except Exception as err:
    bug_dir = os.path.join(args.directory, 'bugs')
    if not os.path.exists(bug_dir):
        os.mkdir(bug_dir)
    this_bug_dir = os.path.join(bug_dir, str(len(os.listdir(bug_dir))))
    os.mkdir(this_bug_dir)
    with open(os.path.join(this_bug_dir, 'cmd.txt'), 'w') as f:
        cmd = ['python', '_run_random_ps.py', f'-d={args.directory}', f'-l={args.length}', f'-s={args.seed}', f'-q={args.sequence}']
        if args.memory_profile:
            cmd.append('-M')
        if args.time_eval:
            cmd.append('-T')
        f.write(' '.join(cmd))
    with open(os.path.join(this_bug_dir, 'err.txt'), 'w') as f:
        f.write(str(err))
    exit(2)
    

# python _run_random_ps.py -d=/home/nie/RelayOpt/out/test -o=0 -s=47 -M -V
# python _run_random_ps.py -d=/home/nie/RelayOpt/out/combine-20230608-032017/7 -l=0 -s=47 -M -T
# python _run_random_ps.py -d=/home/nie/RelayOpt/corpus/_patterns/0 -o=0 -s=47 -M -V
# python _run_random_ps.py -d=/home/nie/RelayOpt/out/test -q=/home/nie/RelayOpt/out/test/single_pass/SimplifyExpr.txt -s=47 -M -V -S