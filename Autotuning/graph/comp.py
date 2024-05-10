import networkx as nx
from typing import Dict

class GraphComparer:
    def __init__(self, abs1:nx.DiGraph, abs2:nx.DiGraph) -> None:
        self.abs1 = abs1
        self.abs2 = abs2

        self.compres = {'diff_nodes':[],'diff_info':''}

    def compare(self):
        abs1_inputs = []
        for n, ndata in self.abs1.nodes.items():
            if str(n).startswith('in'):
                abs1_inputs.append(n)

        abs2_inputs = []
        for n, ndata in self.abs2.nodes.items():
            if str(n).startswith('in'):
                abs2_inputs.append(n)

        if len(abs1_inputs) != len(abs2_inputs):
            self.compres['diff_info'] = 'The numbers of inputs are unequal: ' + str(len(abs1_inputs)) + '/' + str(len(abs2_inputs)) + '.'
            return False
        
        for i in range(len(abs1_inputs)):
            if not self._compare_node(abs1_inputs[i], abs2_inputs[i]):
                return False
        
        return True

    def _compare_node(self, node1:str, node2:str):
        # Compare the content of two nodes
        ndata1 = self.abs1.nodes[node1]
        ndata2 = self.abs2.nodes[node2]

        if not self._compare_dict(ndata1, ndata2):
            self.compres['diff_nodes'] = [(node1, ndata1), (node2, ndata2)]
            self.compres['diff_info'] = 'The contents are different.'
            return False
        
        # Compare the successor nodes
        children1 = []
        for child in self.abs1.successors(node1):
            children1.append(child)
        children2 = []
        for child in self.abs2.successors(node2):
            children2.append(child)

        if len(children1) != len(children2):
            self.compres['diff_nodes'] = [(node1, ndata1), (node2, ndata2)]
            self.compres['diff_info'] = 'The numbers of successors are different: ' + str(len(children1)) + '/' + str(len(children2)) + '.'
            return False
        
        for i in range(len(children1)):
            if not self._compare_node(children1[i], children2[i]):
                return False

        return True

    def _compare_dict(self, dict1:Dict, dict2:Dict):
        if len(dict1) != len(dict2):
            return False
        
        for k in dict1.keys():
            if k not in dict2.keys():
                return False
            if type(dict1[k]) == dict:
                if not self._compare_dict(dict1[k], dict2[k]):
                    return False
            elif type(dict1[k]) == float:
                if not abs(dict1[k] - dict2[k]) < 0.0001:
                    return False
            else:
                if not dict1[k] == dict2[k]:
                    return False
        
        return True
