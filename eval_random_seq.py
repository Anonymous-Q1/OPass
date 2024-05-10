import os
from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64
from tqdm import tqdm
from subprocess import check_output

import tvm
from tvm.relay import transform, parse

from GenCoG_cl.gencog.graph import build_graph
from Autotuning.util import simu_mem_footprint
from Autotuning.sequence import RandomRelaySeq

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-d', '--directory', type=str, default='/home/nie/RelayOpt/out/combine-20230913-224455', help='Case directory.')
    p.add_argument('-s', '--seed', type=int, default=52, help='Random seed of graph generator.')
    p.add_argument('-l', '--len', type=int, default=10, help='Length of generated sequences.')
    p.add_argument('-n', '--num', type=int, default=500, help='# of generated sequences for each case.')
    args = p.parse_args()

default_seq_4 = tvm.ir.transform.Sequential(
    [
        transform.RemoveUnusedFunctions(),
        transform.ToBasicBlockNormalForm(),
        transform.Legalize(),
        transform.SimplifyInference(),
        transform.EliminateCommonSubexpr(),
        transform.CombineParallelConv2D(),
        transform.CombineParallelDense(),
        transform.CombineParallelBatchMatmul(),

        transform.FoldConstant(),
        transform.FoldScaleAxis(),
        transform.SimplifyExpr(),
        transform.CanonicalizeCast(),
        transform.CanonicalizeOps(),
        transform.FlattenAtrousConv(),
        transform.AlterOpLayout(),
        transform.SimplifyExpr(),
        transform.FastMath(),
        transform.FoldConstant(),
        transform.SplitArgs(10),
        transform.FuseOps(4),
    ],
)

def opt_default_level_4(mod):
    with tvm.transform.PassContext(opt_level=4):
        return default_seq_4(mod)

def opt_random_seq(mod, rng):
    seq_generator = RandomRelaySeq(rng)
    relay_seq = seq_generator.generate(max_len=args.len)
    return relay_seq.seq(mod)

def main():
    rng = Generator(PCG64(seed=args.seed))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    for case_dir in os.listdir(args.directory):
        case_path = os.path.join(args.directory, case_dir)
        print('##########################')
        print('Testing', case_path)

        # Test and Record
        tes_res = {}

        filePath = os.path.join(case_path, 'code.txt')
        with open(filePath, 'r') as f:
            mod = parse(f.read())
        mod = transform.InferType()(mod)
        mod = transform.DynamicToStatic()(mod)
        
        # Calculate the memory footprint of the original mod.
        mem = simu_mem_footprint(build_graph(mod)['main'])
        if not isinstance(mem, float):
            continue
        tes_res['Origin'] = {'sim_mem':mem}
        print(f'Origin: {mem} mb.')

        # Calculate the memory footprint of the mod after default optimization.
        default_mod = opt_default_level_4(mod)
        default_mem = simu_mem_footprint(build_graph(default_mod)['main'])
        tes_res['Default'] = {'sim_mem':default_mem}
        print(f'Default: {default_mem} mb.')

        # Calculate the memory footprint of the mod after random optimization.
        best_mem = float('inf')
        for _ in tqdm(range(args.num)):
            try:
                cmd = ['python3', './eval_random_seq_ps.py', f'-p={filePath}', f'-s={rng.integers(2 ** 63)}', f'-l={args.len}']
                r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
                r = str(r, 'utf-8').strip()
                assert r.endswith(' mb')
                random_mem = float(r[:-3])
                if random_mem < best_mem:
                    best_mem = random_mem
            except:
                continue
        tes_res['Random'] = {'sim_mem':best_mem}
        print(f'Random: {best_mem} mb.')


if __name__ == '__main__':
    parse_args()
    main()
