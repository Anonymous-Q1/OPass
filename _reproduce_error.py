import os
from argparse import ArgumentParser
from numpy.random import Generator, PCG64
from Autotuning.debug import ModuleRunner, ModuleError
from Autotuning.sequence import RelaySeq

parser = ArgumentParser()
parser.add_argument('-d', '--directory', default='/home/nie/RelayOpt/out/combine-20230625-053230/14/bugs/10', type=str, help='Error directory.')
args = parser.parse_args()

# Initialize runner
rng = Generator(PCG64(seed=42))
runner = ModuleRunner(rng)

# Load the error information to reproduce
with open(os.path.join(args.directory, 'code.txt'), 'r') as f:
    code = f.read()
with open(os.path.join(args.directory, 'error.txt'), 'r') as f:
    opt_level = int(f.readlines()[0].strip()[-1])
with open(os.path.join(args.directory, 'seq.txt'), 'r') as f:
    seq_info = eval(f.read())
relay_seq = RelaySeq()
relay_seq.from_info(seq_info)

runner.run(code, seq=relay_seq, opt_level=opt_level)
exec_mem, sim_mem = runner.eval_mem(code, seq=relay_seq, opt_level=opt_level)
print(f'{exec_mem:.4f}/{sim_mem:.4f} mb')
exec_time = runner.eval_time(code, seq=relay_seq, opt_level=opt_level)
print(f'{exec_time:.4f} ms')