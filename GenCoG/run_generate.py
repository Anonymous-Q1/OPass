import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError, check_output
from sys import stdout
from time import strftime

from numpy.random import Generator, PCG64
from tqdm import tqdm

from gencog.graph import GraphGenerator, print_relay
from gencog.spec import OpRegistry

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=58, help='Random seed of graph generator.')
    p.add_argument('-o', '--output', type=str, default='out', help='Output directory.')
    p.add_argument('-n', '--num', type=int, default=20, help='Number of generated files.')
    args = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    gen = GraphGenerator(OpRegistry.ops(), rng)
    path = os.path.join(args.output, strftime('run-%Y%m%d-%H%M%S'))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')
    if not os.path.exists(path):
        os.mkdir(path)

    # Generation loop
    progress = tqdm(file=stdout)
    while progress.n < args.num:
        # Generate graph
        graph = gen.generate()
        code = print_relay(graph)

        # Write code to case directory
        case_id = str(progress.n)
        case_path = os.path.join(path, case_id)
        os.mkdir(case_path)
        codePath = os.path.join(case_path, 'code.txt')
        with open(codePath, 'w') as f:
            f.write(code)

        # Evaluate whether this case involves a bug
        cmd = ['python3', '_run_ps.py', f'-d={case_path}', f'-s={rng.integers(2 ** 63)}']
        try:
            run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
        except:
            print('Run failed.')
            os.remove(os.path.join(case_path, 'code.txt'))
            os.rmdir(case_path)
            progress.update()
            continue

        progress.update()


if __name__ == '__main__':
    parse_args()
    main()
