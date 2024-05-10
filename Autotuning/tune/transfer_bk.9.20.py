'''
Use transfer graph to direct the optimization process.
'''

import os
import math
import networkx as nx
from numpy.random import Generator
from typing import Dict, List, Tuple
from tqdm import tqdm
from subprocess import check_output

from Autotuning.util import load_gmod_from_file, simu_mem_footprint
from Autotuning.sequence import RelayPassTable

def _opt_pass_simu_mem(codePath: str, outPath: str, passName: str, seed: int):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join('./', 'python')
    try:
        cmd = ['python3', './_opt_pass_simu_mem.py', f'-i={codePath}', f'-o={outPath}', f'-p={passName}', f'-s={seed}']
        r = check_output(cmd, env=env, timeout=60, stderr=open(os.devnull, 'w'))
        r = str(r, 'utf-8').strip()
        assert r.endswith(' mb')
        random_mem = float(r[:-3])
        return random_mem
    except:
        # print(' '.join(cmd))
        return None

class TranferGraph:
    def __init__(self, codePath: str, rng: Generator) -> None:
        self._rng = rng
        self.code_path_ = codePath

        # The path where the template files are stored.
        self.workspace_ = os.path.join(os.path.dirname(codePath), 'transfer_graph')
        if os.path.exists(self.workspace_):
            os.system(f'rm -rf {self.workspace_}')
        os.mkdir(self.workspace_)

        # Init the work space and transfer graph with target code.
        graph = load_gmod_from_file(self.code_path_)['main']
        status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), simu_mem_footprint(graph))
        assert isinstance(status[-1], float)
        os.system(f'cp {self.code_path_} {os.path.join(self.workspace_, "0.txt")}')

        # The code look-up table. Key: string of code's status. Value: List of code path.
        self.code_lu: Dict[str, List[str]] = {str(status): [os.path.join(self.workspace_, "0.txt")]}
        self.code_num: str = 1

        # The transfer graph
        # Node attributes:
        #   name: str(status)
        #   stat: status
        #   mem: status[-1]
        #   chosen: number of times selected
        #   score: affect the possibility to be chosen
        #   and more... (e.g. a possibility distribution of optimization pass)
        # Edge attribute:
        #   trans: number of transformation along this edge
        self.TG = nx.DiGraph()
        self._add_node(status)

        # Record the transformation path. Key: code path. Value: list of pass name to reach this code.
        self.code_seq: Dict[str, List[str]] = {os.path.join(self.workspace_, "0.txt"): []}

    def run(self, epochs: int):
        self._best_mem = float('inf')
        self._best_mem_epoch = 0
        self._best_mem_code = os.path.join(self.workspace_, "0.txt")
        for epoch in tqdm(range(1, epochs + 1)):
            self._one_iter(epoch)
        print(f'Best mem: {self._best_mem} in {self._best_mem_epoch} epochs by {self.code_seq[self._best_mem_code]}.')

    def _one_iter(self, epoch: int, verbose: bool = True):
        # Choose a computation graph for next optimization
        pick_node = self._node_select()
        pick_code = self._rng.choice(self.code_lu[pick_node])
        self.TG.nodes[pick_node]['chosen'] += 1

        # Choose an optimization pass
        pick_pass = self._pass_select(pick_node)
        if pick_pass is None:
            return
        
        # Optimize the chosen code by the chosen pass
        tmp_path = os.path.join(os.path.dirname(self.code_path_), 'tmp.txt')
        mem = _opt_pass_simu_mem(pick_code, tmp_path, pick_pass, self._rng.integers(2 ** 63))
        # print(pick_node, pick_code, pick_pass, mem)
        if mem is None:
            return

        # Update the workspace
        new_code = os.path.join(self.workspace_, f"{self.code_num}.txt")
        os.system(f'mv {tmp_path} {new_code}')
        self.code_num += 1

        # Update the transfer graph by optimization result, as well as the code_lu and code_seq
        graph = load_gmod_from_file(new_code)['main']
        status = (len(graph.inputs_), len(graph.outputs_), len(graph.oprs_), mem)
        assert isinstance(status[-1], float)
        
        new_node = str(status)
        if new_node not in self.TG:
            print(f'Found new node {new_node} by {new_code}.')
            self._add_node(status)
            self.code_lu[new_node] = []
        self.code_lu[new_node].append(new_code)
        self.code_seq[new_code] = self.code_seq[pick_code] + [pick_pass]

        if (pick_node, new_node) not in self.TG.edges:
            self.TG.add_edge(pick_node, new_node)

        # Update the score of picked node
        # score = e^(-mem)*(trans/chosen)
        trans = len(list(self.TG.successors(pick_node)))
        chosen = self.TG.nodes[pick_node]['chosen']
        old_mem = self.TG.nodes[pick_node]['mem']
        score = math.pow(math.e, - old_mem) * ((trans + 1) / (chosen + 1))
        self.TG.nodes[pick_node]['score'] = score
        # Another choice:
        #       better  unchanged   worse
        # new   inc     inc         keep
        # old   keep    dec         dec
        # ...

        if mem < self._best_mem:
            self._best_mem = mem
            self._best_mem_epoch = epoch
            self._best_mem_code = new_code

        if verbose:
            print(f'Epoch {epoch}: choose {self.TG.nodes[pick_node]["idx"]}, transfer to {self.TG.nodes[new_node]["idx"]}')
            self._print_graph()
    
    def _print_graph(self):
        print('##########')
        for n, ndata in sorted(self.TG.nodes.items(), key=lambda x:x[1]['idx']):
            print(f'Node {ndata["idx"]} {n}: {ndata["mem"]} mb; {ndata["chosen"]} times chosen; \
{len(list(self.TG.successors(n)))} times transfer; score {ndata["score"]}')
        print('##########')

    def _add_node(self, status: Tuple[float]):
        idx = self.TG.number_of_nodes()
        self.TG.add_node(str(status), idx=idx, stat=status, mem=status[-1], 
                         chosen=0, score=1, chosen_pass = [])

    def _node_select(self) -> str:
        p_dict = {}
        for n, ndata in self.TG.nodes.items():
            # TODO: Calculate potential of each node.
            p_dict[n] = ndata['score']
        p_dict = sorted(p_dict.items(), key=lambda e:e[1], reverse=True)    #order by descending
        rand = self._rng.random(dtype='float')
        length = len(p_dict)
        try:
            index = int(math.floor(math.log(math.pow((1-rand),length),0.05)))
        except:
            index = length -1
        if (index>=length -1):
            index = length - 1
        return p_dict[index][0]

    def _pass_select(self, node: str) -> str:
        '''
        Select an optimization pass for a node.
        '''

        # Strategy 0: Randomly selection.
        # pick_pass = self._rng.choice(RelayPassTable.NameTable)

        # Strategy 1: Do not select the passes which have been chosen.
        candidates = [pn for pn in RelayPassTable.NameTable if pn not in self.TG.nodes[node]['chosen_pass']]
        if len(candidates) == 0:
            return None
        pick_pass = self._rng.choice(candidates)
        self.TG.nodes[node]['chosen_pass'].append(pick_pass)

        # Strategy 2: 

        return pick_pass
        
        