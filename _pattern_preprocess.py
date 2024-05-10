import os
from tqdm import tqdm
from numpy.random import Generator, PCG64
from tvm.relay import parser
from Autotuning.util import load_gmod_from_file, exec_and_compare
from Autotuning.pattern import PatternCorpus, PatternReshaper, Pattern
from Autotuning.pattern.reshape import ReshapeError
from GenCoG_cl.gencog.graph.relay import print_relay
from Autotuning.sequence.relay_pass import RelayPassTable

def file_group(root = '/home/nie/RelayOpt/utils/unit_test_patterns/'):
    target = os.path.join(root, '_Pattern')
    if os.path.exists(target):
        os.system(f'rm -rf {target}')
    os.mkdir(target)

    id = 0
    for dn in os.listdir(root):
        if dn == '_Pattern':
            continue
        dp = os.path.join(root, dn)
        if not os.path.isdir(dp):
            continue
        dp = os.path.join(dp, 'Pattern')

        for fn in os.listdir(dp):
            fp = os.path.join(dp, fn)
            tp = os.path.join(target, f'{id}-{dn}-{fn[:-4]}.txt')
            os.system(f'cp {fp} {tp}')
            id += 1

def read_test(root = '/home/nie/RelayOpt/utils/unit_test_patterns/_Pattern/'):
    deleted = []
    for fn in os.listdir(root):
        print('Reading', fn)
        fp = os.path.join(root, fn)
        with open(fp, 'r') as f:
            try:
                _ = parser.parse(f.read())
            except:
                os.system(f'rm -f {fp}')
                deleted.append(fn)
    print('Deleted files:', deleted)
    print('Number of deleted files:', len(deleted))

def build_graph_test(root = '/home/nie/RelayOpt/utils/unit_test_patterns/_Pattern/'):
    for fn in os.listdir(root):
        print('Reading', fn)
        fp = os.path.join(root, fn)
        load_gmod_from_file(fp)

def print_graph_test(root = '/home/nie/RelayOpt/utils/unit_test_patterns/_Pattern/'):
    for fn in os.listdir(root):
        print('Reading', fn)
        fp = os.path.join(root, fn)
        graphs = load_gmod_from_file(fp)
        with open(fp + '.txt', 'w') as f:
            f.write(print_relay(graphs))
        assert exec_and_compare(fp, fp + '.txt')
        os.system(f'rm {fp}.txt')



def rename_patterns(root = '/home/nie/RelayOpt/utils/unit_test_patterns/_Pattern/',
                    target = '/home/nie/RelayOpt/utils/Pattern/'):
    RenameTable = {
        'CanonCast': 'CanonicalizeCast',
        'FakeQuantization': 'FakeQuantizationToInteger',
        'Conv2d': 'CombineParallelConv2D',
        'BatchMatmul': 'CombineParallelBatchMatmul', 
    }

    if os.path.exists(target):
        os.system(f'rm -rf {target}')
    os.mkdir(target)

    i = 0
    for fn in sorted(os.listdir(root)):
        fp = os.path.join(root, fn)
        pn = fn.split("-")[1]
        if pn in RenameTable:
            pn = RenameTable[pn]
        assert pn in RelayPassTable.NameTable, pn
        newfn = f'{i}-{pn}.txt'
        tp = os.path.join(target, newfn)
        os.system(f'cp {fp} {tp}')
        i += 1

def register_patterns(corpus_path = '/home/nie/RelayOpt/corpus/', root = '/home/nie/RelayOpt/utils/Pattern/'):
    rng = Generator(PCG64(seed=42))
    if os.path.exists(corpus_path):
        os.system(f'rm -rf {corpus_path}')
    corpus = PatternCorpus(corpus_path, rng)

    for fn in sorted(os.listdir(root)):
        fp = os.path.join(root, fn)
        passes = [fn[:-4].split('-')[1]]
        graphs = load_gmod_from_file(fp)
        corpus.register(graphs, passes)

def reshape_patterns(corpus_path = '/home/nie/RelayOpt/corpus/'):
    rng = Generator(PCG64(seed=42))
    corpus = PatternCorpus(corpus_path)
    # for i in range(2):   # corpus.size
    #     print('Reshaping', i)
    #     test_reshape_pattern(corpus[i], rng)
    test_reshape_pattern(corpus[165], rng)

def test_reshape_pattern(pattern: Pattern, rng: Generator):
    for _ in tqdm(range(100)):
        rank = rng.integers(0, 6)
        shape = [rng.integers(1, 100) for _ in range(rank)]
        # print(shape)
        try:
            new_gmod = PatternReshaper(pattern, rng).reshape(shape)
            with open(pattern.path_ + '.txt', 'w') as f:
                f.write(print_relay(new_gmod))
        except ReshapeError:
            continue
        except Exception as e:
            print(e)
            raise(e)



if __name__ == '__main__':
    # file_group()
    # read_test()
    # build_graph_test()
    # print_graph_test()
    # rename_patterns()
    # register_patterns()
    reshape_patterns()