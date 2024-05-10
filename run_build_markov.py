from argparse import Namespace, ArgumentParser
from numpy.random import Generator, PCG64

from Autotuning.pattern import MarKovGraph, PatternCorpus

args = Namespace()

def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, default='./', help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=1, help='Random seed of graph generator.')  # 58 exists bug
    p.add_argument('-c', '--corpus', type=str, default='./backup/corpus-6.19', help='Pattern corpus directory.')
    args = p.parse_args()

def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    corpus = PatternCorpus(args.corpus, rng)

    # markov = MarKovGraph(corpus[36], rng)
    # markov.build()
    # markov.load()
    # print('Max Potential:', markov.cal_potential())

    failed_ids = []
    mem_sens_ids = []
    for i in corpus.indices:
        # try:
        if i <= 24:
            continue
        print('###########')
        print(f'Build markov graph for pattern {i}.')
        markov = MarKovGraph(corpus[i], rng, mod='serenity')
        try:
            markov.load()
        except:
            markov.build(max_iter=10)
        max_potential = markov.cal_potential()
        print('Max Potential:', max_potential)
        if max_potential > 0:
            mem_sens_ids.append(i)
        # except:
        #     failed_ids.append(i)
    print('Failed ids:', failed_ids)
    print('Memory sensitive ids:', mem_sens_ids)

if __name__ == '__main__':
    parse_args()
    main()

# Memory sensitive ids: [18, 21, 27, 36, 37]