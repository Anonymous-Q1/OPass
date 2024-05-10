from networkx import DiGraph
from typing import cast, Dict, List, Tuple
from . import GraphAbstractor

def subgraph_match(pattern, graph):
    P:DiGraph = GraphAbstractor('graph').abstract(pattern)
    G:DiGraph = GraphAbstractor('graph').abstract(graph)
    GM = GraphMatcher(P, G)
    return GM.match()

class MatchInfo:
    def __init__(self, match_info:Dict[str, str] = {}) -> None:
        '''
        Map node in pattern to matched node in graph.
        '''
        self.matched_ = match_info     
    
    def add(self, pattern_node:str, graph_node:str):
        if pattern_node in self.matched_:
            assert self.matched_[pattern_node] == graph_node
        else:
            assert graph_node not in self.gnodes
            self.matched_[pattern_node] = graph_node

    @property
    def gnodes(self) -> List[str]:
        '''
        Return all nodes in self.graph_.
        '''
        return list(self.matched_.values())
    
    @property
    def pnodes(self) -> List[str]:
        '''
        Return all nodes in self.pattern_.
        '''
        return list(self.matched_.keys())

    def __str__(self) -> str:
        return str(list(self.matched_.items()))

    def __eq__(self, __value: object) -> bool:
        for k in self.matched_:
            if k not in __value.matched_ or self.matched_[k] != __value.matched_[k]:
                return False
        return True

    def __getitem__(self, idx:str) -> str:
        return self.matched_[idx]
    
    def copy(self):
        new_matched = {}
        for pn in self.matched_:
            new_matched[pn] = self.matched_[pn]
        return MatchInfo(new_matched)

class GraphMatcher:
    def __init__(self, pattern:DiGraph, graph:DiGraph) -> None:
        '''
        Try to find some subgraphs in 'graph' which match pattern.
        '''
        self.pattern_ = pattern
        self.graph_ = graph
    
    def match(self):
        match_order = self._get_match_order()

        matched_subgraphs = []
        matched_pnodes = []
        for n in match_order:
            # Match the first node.
            if n == match_order[0]:
                for node, ndata in self.graph_.nodes.items():
                    if ndata['type'] != 'op':
                        continue
                    if self._match_node(n, node):
                        matched_subgraphs.append(MatchInfo({n:node}))

            else:
                # Find the structures in self.pattern_ to be matched.
                structs = self._get_structures_to_match(n, matched_pnodes)
                # Find the nodes in self.graph_ which match 'n' and the 'structs', then update the matched subgraphs.
                matched_subgraphs = self._match_one_iter(n, structs, matched_subgraphs)
            
            matched_pnodes.append(n)
            if len(matched_subgraphs) == 0:
                return []
        
        # for subg in matched_subgraphs:
        #     print(subg)
        return matched_subgraphs

    def _match_one_iter(self, node:str, structs:List[Tuple[str, str]], matched:List[MatchInfo]):
        '''
        Note that 'node' and 'structs' are in self.pattern_.
        'matched' are maps from nodes in self.pattern_ to self.graph_.
        '''
        # 1. Find all candidate nodes in self.graph_ matched 'node'.
        candidate_gnodes:List[str] = []
        for n, ndata in self.graph_.nodes.items():
            if ndata['type'] != 'op':
                continue
            if self._match_node(node, n):
                assert n not in candidate_gnodes
                candidate_gnodes.append(n)
        
        new_matched = []
        # 2. Check if each 'matched_graph' have any successors matched 'node' and 'structs'.
        for matched_graph in matched:
            matched_graph = cast(MatchInfo, matched_graph)
            cand_gns = candidate_gnodes.copy()

            # 2.1 Filter the candidate gnodes which have already been matched in matched_graph
            _cand_gns = []
            for gn in cand_gns:
                if gn not in matched_graph.gnodes:
                    _cand_gns.append(gn)
            cand_gns = _cand_gns

            # 2.2 Filter the candidate gnodes which cannot match the 'structs'
            for pre_pn, cur_pn in structs:
                assert cur_pn == node

                pre_gn = matched_graph[pre_pn]
                _cand_gns = []
                for cur_gn in cand_gns:
                    if self._match_edge((pre_pn, cur_pn), (pre_gn, cur_gn)):
                        _cand_gns.append(cur_gn)
                cand_gns = _cand_gns
            matched_gns = cand_gns

            # update the graph info
            for gn in matched_gns:
                new_matched_graph = matched_graph.copy()
                new_matched_graph.add(node, gn)
                assert new_matched_graph not in new_matched
                new_matched.append(new_matched_graph)

        return new_matched

    def _get_match_order(self) -> List[str]:
        '''
        Decide the match order of nodes in self.pattern_, the former the earlier.
        Basically conform to the Topological order.
        '''
        # Here we assert the opr node name is ordered by topological order (because of the implementation of abstractor).
        opr_num = 0
        for _, ndata in self.pattern_.nodes.items():
            if ndata['type'] == 'op':
                opr_num += 1
        topo_order = ['opr'+str(i) for i in range(opr_num)]

        # Check if the topo_order is valid.
        gened_tensors = []
        for n in self.pattern_:
            if cast(str, n).startswith('in'):
                assert self.pattern_.nodes[n]['type'] == 'tensor'
                gened_tensors.append(n)
        for n in topo_order:
            # Check if the tensors used by n have already been generated by previous nodes.
            assert n in self.pattern_ and self.pattern_.nodes[n]['type'] == 'op'
            for p in self.pattern_.predecessors(n):
                assert self.pattern_.nodes[p]['type'] == 'tensor'
                assert p in gened_tensors
            for c in self.pattern_.successors(n):
                assert self.pattern_.nodes[c]['type'] == 'tensor'
                gened_tensors.append(c)

        return topo_order
    
    def _get_structures_to_match(self, node:str, matched_nodes:List[str]) -> List[Tuple[str, str]]:
        '''
        Find the structures in self.pattern_ to be matched.
        These structures are connections from some node in 'matched_nodes' to 'node'.
        '''
        connections = []
        for tensor in self.pattern_.predecessors(node):
            assert self.pattern_.nodes[tensor]['type'] == 'tensor'
            for p in self.pattern_.predecessors(tensor):
                assert self.pattern_.nodes[p]['type'] == 'op'
                if p in matched_nodes and (p, node) not in connections:
                    connections.append((p, node))
        return connections
    
    def _match_node(self, pattern_n:str, graph_n:str) -> bool:
        '''
        Check whether pattern_n and graph_n are matched.
        '''
        p_ndata = self.pattern_.nodes[pattern_n]
        g_ndata = self.graph_.nodes[graph_n]

        assert p_ndata['type'] == 'op'
        assert g_ndata['type'] == 'op'

        if p_ndata['op'] != g_ndata['op']:
            return False
        
        # TODO: Match the tensor shape.
        return True
    
    def _match_edge(self, pattern_e:Tuple[str, str], graph_e:Tuple[str, str]) -> bool:
        '''
        Check whether pattern_e and graph_e are matched.
        '''
        assert self.pattern_.nodes[pattern_e[0]]['type'] == 'op'
        assert self.pattern_.nodes[pattern_e[1]]['type'] == 'op'
        assert self.graph_.nodes[graph_e[0]]['type'] == 'op'
        assert self.graph_.nodes[graph_e[1]]['type'] == 'op'

        # Find all tensor nodes connect graph_e[src] to graph_e[des].
        tns_in_ge = []
        for tn in self.graph_.successors(graph_e[0]):
            assert self.graph_.nodes[tn]['type'] == 'tensor'

            for child in self.graph_.successors(tn):
                assert self.graph_.nodes[child]['type'] == 'op'

                if child == graph_e[1] and tn not in tns_in_ge:
                    tns_in_ge.append(tn)
                    break
        
        # Check whether the connection 'graph_e' exists.
        if len(tns_in_ge) == 0:
            return False
        
        # Find all tensor nodes connect pattern_e[src] to pattern_e[des].
        tns_in_pe = []
        for tn in self.pattern_.successors(pattern_e[0]):
            assert self.pattern_.nodes[tn]['type'] == 'tensor'

            for child in self.pattern_.successors(tn):
                assert self.pattern_.nodes[child]['type'] == 'op'

                if child == pattern_e[1] and tn not in tns_in_pe:
                    tns_in_pe.append(tn)
                    break

        # Check if the edge numbers are equal.
        if len(tns_in_pe) != len(tns_in_ge):
            return False

        # TODO: Match the tensor order.
        return True 