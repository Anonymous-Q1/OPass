import os
from numpy.random import Generator, PCG64
from Autotuning.sequence import RelayPassSelector, RelayPassTable, RelaySeq

pass_dir = '/home/nie/RelayOpt/out/test/single_pass/'

def generate_single_pass_files():
    if os.path.exists(pass_dir):
        os.system(f'rm -rf {pass_dir}')
    
    os.mkdir(pass_dir)
    rng = Generator(PCG64(seed=42))
    pass_gen = RelayPassSelector(rng)
    for pass_name in RelayPassTable.NameTable:
        try:
            seq = RelaySeq()
            if pass_name in ['FuseOps', 'CanonicalizeCast']:
                seq.append(pass_gen.wrap_pass('SimplifyInference'))

            p = pass_gen.wrap_pass(pass_name)
            seq.append(p)
            seq.save(os.path.join(pass_dir, pass_name + '.txt'))
        except:
            print(f'Generate pass file "{pass_name}" failed.')

if __name__ == '__main__':
    generate_single_pass_files()