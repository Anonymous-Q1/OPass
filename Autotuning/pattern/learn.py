import os
import json
from typing import List, Dict, cast
from numpy.random import Generator
from collections import defaultdict

from .pattern import PatternCorpus
from Autotuning.sequence import RelayPassTable
from Autotuning.util import compare_code, opt_single_pass
from GenCoG.gencog.graph import Value, Input, Output, Operation, Graph, print_relay
from GenCoG.gencog.graph.base import VertexKind
from GenCoG.gencog.graph.lookup import ValueLookup

class PatternLearner():
    def __init__(self, rng:Generator, corpus:PatternCorpus) -> None:
        self._rng = rng
        self.corpus_ = corpus      # already learned patterns
        self.max_cov = defaultdict(lambda:[0,0])
        
        self.max_cov_file = os.path.join(self.corpus_.path_, 'max_cov.json')
        if os.path.exists(self.max_cov_file):
            with open(self.max_cov_file, 'r') as f:
                cov = json.load(f)
            for k in cov:
                self.max_cov[k][0] = cov[k][0]
                self.max_cov[k][1] = cov[k][1]

    def detect_triggered_pass(self, case_path:str, cov:Dict[str, int]):
        upcov_ps:List[str] = []         # Achieve higher coverage but cannot trigger the pass, i.e., the pass will not modify the graph
        triggered_ps:List[str] = []
        for fn in cov:
            if cov[fn][0] > self.max_cov[fn][0] or cov[fn][1] > self.max_cov[fn][1]:
                print(fn, self.max_cov[fn], '->', cov[fn])
                self.max_cov[fn][0] = max(self.max_cov[fn][0], cov[fn][0])
                self.max_cov[fn][1] = max(self.max_cov[fn][1], cov[fn][1])     

                # check if passes in fn are triggered
                # compare the codes before and after the applied
                for pn in RelayPassTable.SrcTable[fn]:
                    before_opt_path = os.path.join(case_path, 'code.txt')
                    after_opt_path = os.path.join(case_path, pn + '.txt')
                    if not os.path.exists(after_opt_path):
                        print(f'Code file optimized by pass "{pn}" required in {case_path}.')
                        continue

                    if not compare_code(before_opt_path, after_opt_path, self._rng):
                        triggered_ps.append(pn)
                    else:
                        upcov_ps.append(pn)
        
        with open(self.max_cov_file, 'w') as f:
            json.dump(self.max_cov, f, indent='')
        return triggered_ps, upcov_ps
                
    def detect_pattern(self, case_path:str, last_opr:Operation, triggered_ps:List[str]):
        if len(triggered_ps) == 0:
            return {}

        new_patterns:Dict[Graph, List[str]] = defaultdict(list)

        search_list:Dict[int, List[Operation]] = defaultdict(list)
        search_list[0].append(last_opr)
        subg_oprs = []
        visited = [last_opr]
        save_dir = os.path.join(case_path, 'subgraph')
        os.mkdir(save_dir)
        # Incrementally generate subgraph and check whether it triggers the pass.
        while triggered_ps:
            # Check if the seach list is empty. If not, choose the closest node.
            min_d = -1
            for d in search_list.keys():
                if len(search_list[d]) > 0:
                    opr = search_list[d].pop(0)
                    min_d = d
                    break
            if min_d == -1:
                break
            subg_oprs.append(opr)

            # Construct the subgraph and save
            subg = self.constr_subg(subg_oprs)
            save_path = os.path.join(save_dir, f'node#{len(subg_oprs)}.txt')
            code = print_relay(subg)
            with open(save_path, 'w') as f:
                f.write(code)

            # TODO: if the subgraph contain a already learned pattern, then maybe the root cause of pass triggering is such pattern.
            # If so, delete the opr from sub_oprs, and continue.
        
            # Increment the subgraph via Distance-First-Search.
            # Add the neighbour nodes of current vertex to search list.
            for in_value in opr.inputs_:
                in_opr = in_value.def_
                if in_opr.kind != VertexKind.OPR or in_opr in visited:
                    continue
                search_list[min_d + 1].append(in_opr)
                visited.append(in_opr)
            for out_value in opr.outputs_:
                for out_opr in out_value.uses_:
                    if out_opr.kind != VertexKind.OPR or out_opr in visited:
                        continue
                    search_list[min_d + 1].append(out_opr)
                    visited.append(out_opr)
            
            # Check whether the subgraph triggers the pass.
            detected_ps = []
            for pn in triggered_ps:
                out_path = os.path.join(save_dir, f'node#{len(subg_oprs)}-{pn}.txt')
                try:
                    opt_single_pass(save_path, out_path, pn, self._rng)
                    if not compare_code(save_path, out_path, self._rng):
                        new_patterns[subg].append(pn)
                        detected_ps.append(pn)
                except:
                    continue

            # TODO: check if the subgraph triggers the pass without the "last_opr"

            for pn in detected_ps:
                triggered_ps.remove(pn)
        
        for subg in new_patterns:
            pattern = self.corpus_.register(subg,new_patterns[subg])
            print(f'Registered pattern {pattern.idx_} with {len(pattern.graph_.oprs_)} operations to {pattern.path_}, triggering {len(pattern.pass_)} passes:', pattern.pass_)

        return new_patterns

    def constr_subg(self, oprs: List[Operation]) -> Graph:
        '''
        Construct a subgraph based on a given operation set.
        '''
        
        values:List[Value] = []
        for opr in oprs:
            for value in opr.outputs_ + opr.inputs_:
                if value not in values:
                    values.append(value)
        
        # Copy values to protect the old graph
        new_values:List[Value] = []
        value_map:Dict[Value, Value] = {}
        param_idc:Dict[Value, bool] = defaultdict(lambda :False)    # indicate if a value is a parameter
        for value in values:
            new_value = Value(value.type_)
            new_values.append(new_value)
            value_map[value] = new_value

            if value.def_.kind == VertexKind.IN:
                value.def_ = cast(Input, value.def_)
                param_idc[new_value] = value.def_.is_param_
            
            for opr in value.uses_:
                if opr.kind == VertexKind.OUT:
                    continue
                opr = cast(Operation, opr)
                value_idx = opr.inputs_.index(value)
                if len(opr.op_.params_) > 0 and value_idx in opr.op_.params_:
                    assert param_idc[new_value] == True
        
        # Copy oprs and connect them with copied values
        new_oprs = []
        for opr in oprs:
            opr_inputs = [value_map[v] for v in opr.inputs_]
            opr_outputs = [value_map[v] for v in opr.outputs_]
            new_oprs.append(Operation(opr.op_, opr.attrs_, opr_inputs, opr_outputs))
        
        outputs = [Output(v) for v in new_values if len(v.uses_) == 0]
        inputs = [Input.from_value(v, param_idc[v]) for v in new_values if v.def_ is None]
        graph = Graph(inputs, outputs, new_oprs)
        return graph
        