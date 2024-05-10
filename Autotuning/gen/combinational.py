from typing import Iterable, List, Optional, Dict, cast
from collections import defaultdict
from numpy.random import Generator
import numpy as np
from tvm.relay import parse
from GenCoG_cl.gencog.graph import Value, Graph, Output, GraphMod, print_relay, build_graph, Operation
from GenCoG_cl.gencog.solve.solver import RelayTypeKind, TupleTensorType
from GenCoG_cl.gencog.graph.lookup import ValueLookup
from GenCoG_cl.gencog.spec import OpRegistry
from ..solve import TypeSolverE, ShapeCustomer
from ..pattern import Pattern
from ..graph import GraphUtil, GraphVisitorForGen
from ..config import DEBUG, PARAMS
from ..pattern import PatternCorpus, ReshapeError, reshaper

max_rank = 5
max_dim = 100
max_trial = 1000
use_penal = 4

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)

class CombGraphGenerator:
    '''
    Combine several patterns to a graph.
    '''
    def __init__(self, corpus: PatternCorpus, rng:Generator) -> None:
        self._rng = rng
        self.corpus_ = corpus
        
    def generate(self, max_pattern_num: int = 5):
        # For Debug:
        self.combined_pids = []

        # Sample an initial pattern and init the graph.
        gmod = self._init_graph()
        graph = gmod['main']

        value_lu = ValueLookup()
        for v in GraphVisitorForGen().abstract_values(graph):
            value_lu.add(v)

        # Sample patterns and combine them to the graph.
        pattern_num = 1
        trial = 0
        while pattern_num < max_pattern_num:
            trial += 1
            if trial >= max_trial:
                break

            pattern = self._sample_pattern()

            # TODO: Current mode: do not append zero-input patterns and variable-type patterns.
            if pattern.rule_ is None:
                continue
            # print(f'Trying to append pattern {pattern.idx_}, target rank range: {pattern.rule_.reshapeR_.inRs_[pattern.rule_.reshapeR_.anchor_idx].rank_range_}...')
            
            # Try to sample a value as the connection point
            cp_value = self._sample_connect_point(pattern, value_lu, {})
            if cp_value is None:
                continue
            # print(f'Founded connection value with rank {cp_value.type_.rank}, dtype {cp_value.type_.dtype_}.')
            
            # Try to reshape the pattern conforming the sampled anchor value.
            try:
                re_gmod = reshaper(pattern, cp_value.type_.shape_, self._rng)
            except ReshapeError:
                continue
            except Exception as e:
                raise e
            # print('Successfully reshape.')

            # Connect the reshaped pattern to the graph.
            # TODO: Consider when gmod contains several function.
            assert len(re_gmod.funcNames) == 1 and 'main' in re_gmod.funcNames
            self._append_pattern_graph(re_gmod['main'], graph, pattern.rule_.reshapeR_.anchor_idx, cp_value, value_lu)
            pattern_num += 1
            trial = 0

            # For Debug
            self.combined_pids.append(pattern.idx_)
        
        # Compose the outputs to one output tuple.
        out_values = []
        out_types = []
        for out in graph.outputs_:
            out_values.append(out.value_)
            out_types.append(out.value_.type_)
            out.value_.uses_.remove(out)
            out.value_ = None
        output = Value(TupleTensorType(out_types))
        opr = Operation(OpRegistry.get('tuple'), [], out_values, [output])
        graph.oprs_.append(opr)
        graph.outputs_ = [Output(output)]

        # For Debug
        print(self.combined_pids)
        return gmod

    def _sample_pattern(self) -> Pattern:
        return self._rng.choice(list(self.corpus_.patterns_.values()))

    def _init_graph(self) -> GraphMod:
        init_pattern = self._sample_pattern()
        # print(f'Trying to initialize graph with pattern {init_pattern.idx_}...')
        # TODO: finetune the shape of initial pattern.
        # TODO: Current mode: do not append zero-input patterns and variable-type patterns.
        while True:
            if init_pattern.rule_ is None:
                init_pattern = self._sample_pattern()
            else:
                break
        
        # For Debug
        self.combined_pids.append(init_pattern.idx_)

        with open(init_pattern.path_, 'r') as f:
            return build_graph(parse(f.read()))

    def _append_pattern_graph(self, pg: Graph, graph: Graph, anchor_idx: int, cp_value: Value, value_lu: ValueLookup):
        # Append the main function graph of reshaped pattern to graph.
        # The connection point is between cp_value of graph and anchor_input of pattern.
        available_values = GraphVisitorForGen().abstract_values(pg)
        available_values.remove(pg.inputs_[anchor_idx].value_)

        # If cp_value is an output value, then delete the output vertex.
        for out in cp_value.uses_:
            if isinstance(out, Output):
                out.value_ = None
                graph.outputs_.remove(out)
        cp_value.uses_ = [vtx for vtx in cp_value.uses_ if not isinstance(vtx, Output)]

        # Connect the anchor input.
        anchor_in = pg.inputs_[anchor_idx]
        assert anchor_in.value_.type_.shape_ == cp_value.type_.shape_
        for vtx in anchor_in.value_.uses_:
            if isinstance(vtx, Operation):
                vtx.inputs_ = [cp_value if v is anchor_in.value_ else v for v in vtx.inputs_]
            elif isinstance(vtx, Output):
                assert vtx.value_ == anchor_in.value_
                vtx.value_ = cp_value
            else:
                raise Exception
        cp_value.uses_ += anchor_in.value_.uses_
        anchor_in.value_.uses_ = []

        # Append other inputs to the graph.
        # TODO: Sample values from value_lu to connect other inputs
        graph.inputs_ += [inp for inp in pg.inputs_ if inp != anchor_in]

        # Append oprs, outrefs, outputs and typevars.
        graph.oprs_ += pg.oprs_
        graph.outrefs_ += pg.outrefs_
        graph.typevars_ += pg.typevars_
        graph.outputs_ += pg.outputs_

        # Append values to value_lu.
        for v in available_values:
            value_lu.add(v)

    def _sample_value(self, values: List[Value], add_cnt: Dict[Value, int]):
        num_uses = [len(v.uses_) + add_cnt.get(v, 0) for v in values]
        scores = softmax(-use_penal * np.array(num_uses, dtype='float32'))
        return self._rng.choice(values, p=scores)
    
    def _check_static_shape(self, value: Value) -> bool:
        # If values's shape is (T.Any(), ..), then return False.
        if value.type_.rank == 0:
            return True
        for dim in value.type_.shape_:
            if not isinstance(dim, int):
                return False
        return True

    def _sample_connect_point(self, pattern: Pattern, value_lu: ValueLookup, add_cnt:Dict[Value, int]) -> Value:
        # Sample a value from graph as a connection point, based on the rank of the pattern's anchor input
        anchor_idx = pattern.rule_.reshapeR_.anchor_idx
        anchor_rank_range = pattern.rule_.reshapeR_.inRs_[anchor_idx].rank_range_
        anchor_dtype = pattern.gmod_['main'].inputs_[anchor_idx].value_.type_.dtype_

        rank_choices = [r for r in range(anchor_rank_range[0], anchor_rank_range[1] + 1)]
        dtype_choices = [anchor_dtype]

        matches = list(value_lu.by_choices(rank_choices, dtype_choices))
        matches = [v for v in matches if self._check_static_shape(v)]
        return None if len(matches) == 0 else self._sample_value(matches, add_cnt)