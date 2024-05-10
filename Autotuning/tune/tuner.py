'''
The main tuning process of sequence customization
'''
import os
import networkx as nx
from numpy.random import Generator
from collections import defaultdict
from typing import Dict, cast
from ..pattern import PatternCorpus, MarKovGraph
from ..sequence import RelayPassSelector, RelaySeq

class Tuner:
    def __init__(self, corpus: PatternCorpus, rng:Generator) -> None:
        self.rng_ = rng
        self.corpus_ = corpus
        self.RPS = RelayPassSelector(self.rng_)

    def tune(self, casePath:str, max_step = 10):
        '''
        Tune the code of 'casePath\code.txt'
        '''

        # 1.1 Recognize patterns in codePath
        # 1.2 Calculate the frequency and scaling rate of each pattern.
        # Currently we just assume we've already known the patterns.
        # recog_patterns = {}
        # for dn in os.listdir(casePath):
        #     pattern_path = os.path.join(casePath, dn)
        #     if os.path.isdir(pattern_path):
        #         try:
        #             pattern_id = int(dn)
        #         except:
        #             continue
        #         recog_patterns[pattern_id] = {'freq':1, 'scale':1, 'pattern':self.corpus_[pattern_id]}
        recog_patterns = self._recognize_patterns(casePath)

        # 2 Predict the memory potential
        # 2.0 Load the markov graph if exists, or build it
        for pid in recog_patterns:
            markov = MarKovGraph(recog_patterns[pid]['pattern'], self.rng_)
            markov.load()
            markov.cal_potential()
            recog_patterns[pid]['markov'] = markov.markovGraph
            
            recog_patterns[pid]['cursor'] = '0'
        
        acc_reward = 0
        customized_seq = RelaySeq()
        for step in range(max_step):
            print('step', step)
            # 2.1 Predict the potential of each pattern
            potentials = self._pred_P(recog_patterns)
            # 2.2 Choose the pass with maximum potential
            chosen_pass, max_potential = self._choose_pass(potentials)
            print(chosen_pass, max_potential)
            if max_potential <= 0:
                break
            # 2.3 Apply the chosen pass and move the cursor
            reward = self._move_cursor(chosen_pass, recog_patterns)
            acc_reward += reward
            print('reward', reward)
            print('acc reward', acc_reward)

            if chosen_pass in ['FuseOps', 'CanonicalizeCast']:
                precond_pass = self.RPS.wrap_pass('SimplifyInference')
                if not customized_seq.contained(precond_pass, param_compared=False):
                    customized_seq.append(precond_pass)
            customized_seq.append(self.RPS.wrap_pass(chosen_pass))
        return customized_seq

    def _choose_pass(self, pass2potential: Dict[str, float]):
        max_potential = sorted(pass2potential.items(), key=lambda x:x[1], reverse=True)[0][1]
        candicates = [pn for pn in pass2potential if pass2potential[pn] == max_potential]
        # print(sorted(pass2potential.items(), key=lambda x:x[1], reverse=True))
        return str(self.rng_.choice(candicates)), max_potential
    
    def _recognize_patterns(self, casePath:str):
        if casePath == '/home/nie/RelayOpt/out/combine-20230626-232349/16':
            recog_patterns = {
                18:{'freq': 1, 'scale': 40000, 'pattern':self.corpus_[18]},
                27:{'freq': 1, 'scale': 20313, 'pattern':self.corpus_[27]},
                36:{'freq': 1, 'scale': 2666, 'pattern':self.corpus_[36]},
            }
            return recog_patterns
        
        raise Exception('Cannot recognize patterns from', casePath)
    
    def _pred_P(self, patterns_info:Dict):
        pass2potential = defaultdict(lambda:0)
        for pid in patterns_info:
            pattern_P = self._pred_pattern_P(patterns_info[pid])
            # print(pid, pattern_P)
            for pn in pattern_P:
                pass2potential[pn] += pattern_P[pn]
        return pass2potential

    def _pred_pattern_P(self, pattern_info:Dict):
        pass2potential:Dict[str, float] = {}
        markovG = cast(nx.DiGraph, pattern_info['markov'])
        cursor_node = pattern_info['cursor']
        for n in markovG.successors(cursor_node):
            for pass_name in markovG[cursor_node][n]['passName']:
                reward = float(markovG[cursor_node][n]['reward'])
                potential = float(markovG.nodes[n]['potential'])
                pass2potential[pass_name] = (reward + potential) * pattern_info['freq'] * pattern_info['scale']
        return pass2potential
    
    def _move_cursor(self, pass_name:str, patterns_info:Dict):
        reward = 0
        for pid in patterns_info:
            markovG = cast(nx.DiGraph, patterns_info[pid]['markov'])
            cursor_node = patterns_info[pid]['cursor']
            for n in markovG.successors(cursor_node):
                if pass_name in markovG[cursor_node][n]['passName']:
                    patterns_info[pid]['cursor'] = n
                    break
            new_cursor = patterns_info[pid]['cursor']
            if cursor_node != new_cursor:
                rate = patterns_info[pid]['freq'] * patterns_info[pid]['scale']
                reward += float(markovG[cursor_node][new_cursor]['reward']) * rate
            # else:
            #     raise Exception(f'Cannot move cursor node {cursor_node} with pass {pass_name}.')
            print(f'pattern {pid}: {cursor_node}->{new_cursor}')
        return reward