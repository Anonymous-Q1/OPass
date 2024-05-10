import networkx as nx
from networkx.drawing.nx_pydot import write_dot, read_dot
from numpy.random import Generator
import matplotlib.pyplot as plt
import os
from typing import List, Union, cast
from tvm.error import TVMError

from ..util import compare_graph, opt_single_pass, viz2file, simu_mem_footprint, load_graph_from_file, code_valid_check
from ..serenity import simu_mem_serenity
from . import Pattern
from ..sequence import RelayPassTable

class MarKovGraph:
    def __init__(self, pattern:Pattern, rng:Generator, mod:str = 'static') -> None:
        self.init_state_ = pattern
        self.dir_ = os.path.join(os.path.dirname(pattern.path_), 'markov')
        self.rng_ = rng

        if mod == 'static':
            self.memory_evaluator = simu_mem_footprint
        elif mod == 'serenity':
            self.memory_evaluator = lambda x:simu_mem_serenity(x, time_limit=60)
        else:
            raise Exception(f'No memory evaluator named {mod}.')

    def build(self, max_iter = 20):
        self.markovGraph = nx.DiGraph()
        self.states = {}
        self.unexplored = []
        if os.path.exists(self.dir_):
            os.system(f'rm -rf {self.dir_}')
        os.mkdir(self.dir_)

        # Initialize the state.
        # Copy the pattern file to '0' case path
        init_state_path = os.path.join(self.dir_, '0')
        os.mkdir(init_state_path)
        os.system(f'cp {self.init_state_.path_} {init_state_path}')
        # Visualization
        viz2file(os.path.join(init_state_path, 'code.txt'))
        # Similate memory footprint
        mem = self.memory_evaluator(self.init_state_.graph_)
        # Initialize graph, states, and unexplored ids.
        self._add_node_to_markov(0, mem)
        self.states[0] = {'path': init_state_path, 'graph': self.init_state_.graph_, 'memory': mem}
        print(f'Initiate graph 0 with memory footprint {mem} mb.')
        self.unexplored.append(0)

        # Construct the markov graph.
        iter = 0
        while self.unexplored:
            # Choose an unexplored node to further explore.
            id = self.unexplored.pop(0)
            self._explore_one_node(id, valid_check=True, prune=True)
            
            iter += 1
            if iter >= max_iter:
                break

    def load(self):
        '''
        Load markov graph from saved file.
        '''
        graph_path = os.path.join(self.dir_, 'file.dot')
        assert os.path.exists(graph_path), f'Markov graph save file {graph_path} not exists.'

        self.markovGraph = nx.DiGraph(read_dot(graph_path))
        self.markovGraph.remove_node('\\n')
        for _, ndata in self.markovGraph.nodes.items():
            ndata['memory'] = ndata['memory'].strip('"')
        for _, edata in self.markovGraph.edges.items():
            edata['reward'] = edata['reward'].strip('"')
            # edata['passName'] = edata['passName'].strip('"')
            edata['passName'] = eval(edata['passName'].strip('"'))
    
    def cal_potential(self):
        '''
        Calculate the potential of every vertex in markov graph.
        '''
        footprints = {}
        for n, ndata in self.markovGraph.nodes.items():
            footprints[n] = float(ndata['memory'])
        
        while len(footprints) > 0:
            sort_by_mem = sorted(footprints.items(), key=lambda x:x[1])
            lowest_node = sort_by_mem[0][0]
            update_bound = sort_by_mem[0][1]

            to_update_list = [lowest_node]
            while to_update_list:
                to_update = to_update_list.pop(0)
                footprints.pop(to_update)
                self.markovGraph.nodes[to_update]['bound'] = str(update_bound)
                for n in self.markovGraph.predecessors(to_update):
                    if n in footprints and n not in to_update_list:
                        to_update_list.append(n)

        for n, ndata in self.markovGraph.nodes.items():
            ndata['potential'] = str(max(0, float(ndata['memory']) - float(ndata['bound'])))
        
        return float(self.markovGraph.nodes['0']['memory']) - float(self.markovGraph.nodes['0']['bound'])
        
            
    def _explore_one_node(self, id, valid_check = False, prune = False):
        path = self.states[id]['path']
        codePath = os.path.join(path, 'code.txt')
        origin_mem = self.states[id]['memory']
        # graph = self.states[id]['graph']

        # Optimize the chosen graph by every pass.
        tmpCodePath = os.path.join(self.dir_, 'tmpCode.txt')
        for passName in RelayPassTable.NameTable:
            try:
                opt_single_pass(codePath, tmpCodePath, passName, self.rng_)
            except Exception as _:
                self._add_node_to_markov('bug', float('inf'))
                self._add_edge_to_markov(id, 'bug', -float('inf'), passName)
                continue

            # Check if this new graph is already exist in Markov graph.
            try:
                newGraph = load_graph_from_file(tmpCodePath, self.rng_)
                found_id = self._find(newGraph)
            except TVMError as _:
                newGraph = None
                found_id = -1

            # If not exist.
            if found_id == -1 :
                # Save and visualize the graph.
                newid = len(self.states)
                newPath = os.path.join(self.dir_, str(newid))
                os.mkdir(newPath)
                newCodePath = os.path.join(newPath, 'code.txt')
                os.system(f'cp {tmpCodePath} {newCodePath}')
                with open(os.path.join(newPath, f'{passName}-from-{id}'), 'w') as f:
                    pass

                # Check if the newly generated code is valid
                if valid_check and not code_valid_check(newCodePath, rng=self.rng_):
                    print(f'Graph {newid} cannot run normally.')
                    self._save_bug(newid)

                    self._add_node_to_markov('bug', float('inf'))
                    self._add_edge_to_markov(id, 'bug', -float('inf'), passName)
                    continue

                viz2file(newCodePath)

                # Simulate the memory footprint
                try:
                    mem = self.memory_evaluator(newGraph)
                    reward = origin_mem - mem
                except:
                    print(f'Graph {newid}\'s memory footprint cannot be simulated.')
                    with open(os.path.join(newPath, 'SIMUBUG'), 'w') as f:
                        pass
                    self._save_bug(newid)

                    self._add_node_to_markov('bug', float('inf'))
                    self._add_edge_to_markov(id, 'bug', -float('inf'), passName)
                    continue

                # Update states, graph and unexplored
                self.states[newid] = {'path': newPath, 'graph': newGraph, 'memory': mem}
                self.unexplored.append(newid)
                self._add_node_to_markov(newid, mem)
                self._add_edge_to_markov(id, newid, reward, passName)

                with open(os.path.join(newPath, f'R{reward:.4g}-M{mem:.4g}'), 'w') as f:
                    pass
                
                print(f'Found new graph {newid} with memory footprint {mem} mb / reward {reward}. Generated from {id}.')
                print(f'Add edge {id}->{newid}, pass {passName}.')

                # Prune the node if the rewards from its parent and grandparent are all negative
                if prune and reward <= 0 and id != 0:
                    need_prune = True
                    for pnode in self.markovGraph.predecessors(str(id)):
                        if float(self.markovGraph[pnode][str(id)]['reward']) > 0:
                            need_prune = False
                            break
                    if need_prune:
                        self.unexplored.pop()
                        print(f'Graph {newid} is pruned.')
                
                self._save_graph_struct()

            # If exist.
            # elif id != found_id:
            else:
                mem = self.states[found_id]['memory']
                self._add_edge_to_markov(id, found_id, origin_mem - mem, passName)
                self._save_graph_struct()
                print(f'Add edge {id}->{found_id}, pass {passName}.')
                   
    def _save_graph_struct(self):
        write_dot(self.markovGraph, os.path.join(self.dir_, 'file.dot'))

        plt.figure(figsize=(15, 10))
        plt.cla()
        G = self.markovGraph
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos=pos, with_labels=True, 
                 labels={n:'%s-%.3g'%(n, float(nd['memory'])) for n, nd in G.nodes.items()})
        # nx.draw_networkx_edge_labels(G, pos=pos,
        #          edge_labels={e:'%.3g'%(float(ed['reward'])) for e, ed in G.edges.items()})
        plt.savefig(os.path.join(self.dir_, 'markov.png'))
        plt.close()
    
    def _save_bug(self, id):
        codePath = os.path.join(self.dir_, str(id))
        bugPath = os.path.join(self.dir_, 'bugs')
        if not os.path.exists(bugPath):
            os.mkdir(bugPath)
        bugPath = os.path.join(bugPath, str(len(os.listdir(bugPath))))
        if not os.path.exists(bugPath):
            os.mkdir(bugPath)
        
        os.system(f'mv {codePath}/* {bugPath}')
        os.rmdir(codePath)
        
    def _find(self, g):
        '''
        Find same graph in already generated graphs.
        If found, return corresponding id.
        Else, return -1.
        '''
        for id in self.states:
            if compare_graph(g, self.states[id]['graph']):
                return id
        return -1
            
    def _add_node_to_markov(self, name:Union[int, str], memory:Union[float, str]):
        name = str(name)
        memory = str(memory)
        if name not in self.markovGraph.nodes:
            self.markovGraph.add_node(name, memory=memory)
        else:
            assert self.markovGraph.nodes[name]['memory'] == memory

    def _add_edge_to_markov(self, src:Union[int, str], des:Union[int, str], reward:Union[float, str], passName:str):
        src = str(src)
        des = str(des)
        reward = str(reward)
        if (src, des) not in self.markovGraph.edges:
            self.markovGraph.add_edge(src, des, reward=reward, passName = [passName])
        else:
            assert self.markovGraph[src][des]['reward'] == reward
            if passName not in self.markovGraph[src][des]['passName']:
                cast(List, self.markovGraph[src][des]['passName']).append(passName)




# import networkx as nx
# from networkx.drawing.nx_pydot import write_dot, read_dot
# G = nx.DiGraph()
# G.add_node(1, n = 1)
# G.add_node(2, n = 2)
# G.add_edge(2, 1, e = '12')
# write_dot(G, 'file.dot')

# G:nx.DiGraph = read_dot('file.dot')
# for n, nd in G.nodes.items():
#     print(n, nd)
# for e, ed in G.edges.items():
#     print(e, ed)