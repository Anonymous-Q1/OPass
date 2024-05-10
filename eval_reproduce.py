import os
from subprocess import check_output
import json

from Autotuning.util import viz2file, simu_mem_from_relay, cal_tvm_mem, serenity_mem_from_relay, load_gmod_from_file
from Autotuning.serenity_test import simu_mem_serenity_test
from tvm.relay import parse, transform

def _opt_pass_simu_mem(codePath: str, outPath: str, passName: str, seed: int, profiler):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join('./', 'python')

    if profiler == simu_mem_from_relay:
        profiler_name = 'static'
    elif profiler == cal_tvm_mem:
        profiler_name = 'tvm'
    elif profiler == serenity_mem_from_relay:
        profiler_name = 'serenity'
    else:
        raise Exception(f'No such profiler {profiler}')

    try:
        cmd = ['python3', './_opt_pass_simu_mem.py', f'-i={codePath}', f'-o={outPath}', f'-p={passName}', f'-s={seed}', f'-m={profiler_name}']
        r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
        r = str(r, 'utf-8').strip()
        assert r.endswith(' mb')
        random_mem = float(r[:-3])
        return random_mem
    except:
        print(' '.join(cmd))
        return None
    
def reproduce(case_path: str, mode: str = 'static'):
    code_path = os.path.join(case_path, 'code.txt')
    with open(code_path, 'r') as f:
        mod = parse(f.read())
    mod = transform.InferType()(mod)
    mod = transform.DynamicToStatic()(mod)
    code_path += '.txt'
    with open(code_path, 'w') as f:
        f.write(mod.astext())
    
    if mode == 'static':
        res_path = os.path.join(case_path, 'tune_results.json')
        profiler = simu_mem_from_relay
    elif mode == 'dynamic':
        res_path = os.path.join(case_path, 'tune_results_serenity.json')
        profiler = serenity_mem_from_relay
    else:
        raise Exception(mode)
    
    with open(res_path, 'r') as f:
        res = json.load(f)
    seq = eval(res['Transfer'][1])

    print(seq)
    # Reproduce
    rep_path = os.path.join(case_path, 'reproduce')
    if os.path.exists(rep_path):
        os.system(f'rm -rf {rep_path}')
    os.mkdir(rep_path)
    for idx, pn in enumerate(seq):
        new_code_path = os.path.join(rep_path, f'{idx}.txt')
        mem = _opt_pass_simu_mem(code_path, new_code_path, pn, 42, profiler)
        viz2file(new_code_path)
        code_path = new_code_path
        print(idx, pn, mem)

def main():
    case_path = '/home/nie/RelayOpt/out/dynamic/combine-20231006-231536-5/'
    reproduce(case_path, 'dynamic')

    # rep_path = os.path.join(case_path, 'reproduce')
    # code_path = os.path.join(rep_path, '2.txt')
    # gmod = load_gmod_from_file(code_path)
    # simu_mem_serenity_test(gmod['main'])

    # rep_path = os.path.join(case_path, 'reproduce (copy)')
    # code_path = os.path.join(rep_path, '1.txt')
    # gmod = load_gmod_from_file(code_path)
    # simu_mem_serenity_test(gmod['main'])

if __name__ == '__main__':
    main()