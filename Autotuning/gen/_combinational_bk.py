from typing import Iterable, List, Optional, Dict, cast
from collections import defaultdict
from numpy.random import Generator
from GenCoG.gencog.graph import GraphGenerator, Operation, Input
from GenCoG.gencog.solve.store import ValueStore, StoreNode, ScalarNode, ArrayNode
from GenCoG.gencog.spec import Op, int_expr_choices, expr_choices
from GenCoG.gencog.graph.lookup import ValueLookup
from GenCoG.gencog.config import params
from GenCoG.gencog.graph.base import Graph, Output, Value
from GenCoG.gencog.expr import DataType
from GenCoG.gencog.expr.ty import common_dtypes
from GenCoG.gencog.solve import TypeSolver, TensorType
from ..solve import TypeSolverE, ShapeCustomer
from ..pattern import Pattern
from ..graph import GraphUtil
from ..config import DEBUG, PARAMS

max_rank = PARAMS['max_rank']
max_dim = PARAMS['max_dim']
pattern_trial = PARAMS['pattern_trial']

class CombGraphGenerator(GraphGenerator):
    def __init__(self, patterns:List[Pattern], ops:Iterable[Op], rng:Generator) -> None:
        super().__init__(ops, rng)
        self._patterns = patterns
        self.SC = ShapeCustomer(self._rng)

    def generate(self, max_pattern_num:int = 5, pattern_ids:Optional[List[int]] = None) -> Graph:
        # if DEBUG:
        #     patterns = [p for p in self._patterns if p.idx_ in pattern_ids]
        #     print('Choosed patterns:', [p.idx_ for p in patterns])
        #     return self.combine(patterns)
        combined_ids = []

        # Initial generation
        graph, pattern = self._sample_first_pattern()
        combined_ids.append(pattern.idx_)

        value_lu = ValueLookup()
        for opr in graph.oprs_:
            for value in opr.inputs_ + opr.outputs_:
                if value not in value_lu.values:
                    value_lu.add(value)
        for out in graph.outputs_:
            out.value_.uses_.remove(out)
        inputs = graph.inputs_
        oprs = graph.oprs_
        dtype = inputs[0].value_.type_.dtype_
        
        pattern_num = 1
        while(pattern_num < max_pattern_num):
            # Sample a pattern
            pattern = self.random_choice_patterns(1)[0]
            # print('Try to append pattern', pattern.idx_)

            for _ in range(pattern_trial):
                # Try to sample a value according to patterns' input
                subg = GraphUtil.copy(pattern.graph_)
                self._finetune_dtype(subg, dtype)

                sampled_input = self._rng.choice(subg.inputs_)
                sampled_input = cast(Input, sampled_input)
                sampled_value = self._sample_connect_point(sampled_input.value_, value_lu, {})
                if sampled_value is None:
                    continue

                # Try to finetune the pattern
                try:
                    self._finetune_shape(subg, sampled_input.value_, sampled_value.type_.shape_)
                except:
                    continue

                # Connect the pattern to graph
                # connect inputs
                # TODO: sample values from value_lu to connect other inputs
                inputs += [inp for inp in subg.inputs_ if inp is not sampled_input]
                for vtx in sampled_input.value_.uses_:
                    assert isinstance(vtx, Operation)
                    vtx.inputs_ = [v if v != sampled_input.value_ else sampled_value for v in vtx.inputs_]
                sampled_value.uses_ += sampled_input.value_.uses_
                sampled_input.value_.uses_ = []

                # add oprs
                oprs += subg.oprs_

                # delete outputs
                for out in subg.outputs_:
                    out.value_.uses_.remove(out)

                # add values
                for opr in subg.oprs_:
                    for v in opr.inputs_ + opr.outputs_:
                        if v not in value_lu.values:
                            value_lu.add(v)

                pattern_num += 1
                combined_ids.append(pattern.idx_)
                # print('Successfully append pattern', pattern.idx_)
                break
        
        outputs = [Output(v) for v in value_lu.values if len(v.uses_) == 0]
        return Graph(inputs, outputs, oprs), combined_ids
        
    def random_choice_patterns(self, n:int) -> List[Pattern]:
        return self._rng.choice(self._patterns, n)
    
    def _sample_first_pattern(self):
        while True:
            pattern = self.random_choice_patterns(1)[0]
            graph = GraphUtil.copy(pattern.graph_)
            
            # try:
            print(pattern.idx_)
            sampled_input = cast(Input, self._rng.choice(graph.inputs_))
            old_shape = sampled_input.value_.type_.shape_
            new_shape = [dim*200 if d_idx > len(old_shape)-3 else dim for d_idx, dim in enumerate(old_shape)]
            self._finetune_shape(graph, sampled_input.value_, new_shape)
            # except:
            #     continue

            # print('Initial graph with pattern', pattern.idx_)

            return graph, pattern

    def combine(self, patterns:List[Pattern]) -> Graph:
        inputs:List[Input] = []
        oprs:List[Operation] = []
        value_lu = ValueLookup()
        outputs:List[Output] = []

        dtype:DataType = None
        for idx, pattern in enumerate(patterns):
            # copy the graph to avoid pollution
            subg = GraphUtil.copy(pattern.graph_)

            # finetuning the dtype of subg to unify the dtypes
            if dtype is None:
                dtype = subg.inputs_[0].value_.type_.dtype_
            else:
                self._finetune_dtype(subg, dtype)
            
            # finetuning the shape if the subg is the first one
            if idx == 0:
                fitted_g = None
                input_num = len(subg.inputs_)
                for in_idx in range(input_num):
                    new_subg = GraphUtil.copy(subg)
                    old_shape = new_subg.inputs_[in_idx].value_.type_.shape_
                    new_shape = [dim*3 if d_idx > len(old_shape)-5 else dim for d_idx, dim in enumerate(old_shape)]
                    try:
                        self._finetune_shape(new_subg, new_subg.inputs_[in_idx].value_, new_shape)
                        fitted_g = new_subg
                        break
                    except:
                        pass
                if fitted_g is None:
                    raise Exception('Cannot fit pattern', pattern.idx_)
                subg = fitted_g
            return subg

            # combine the subg to the whole graph
            # r = self._combine_subg(subg, inputs, oprs, value_lu)
            # if not r:
            #     return None
            
            break
            
        outputs = [Output(v) for v in value_lu.values if len(v.uses_) == 0]
        return Graph(inputs, outputs, oprs)
            
    def _combine_subg(self, graph:Graph, inputs:List[Input], 
                      oprs:List[Operation], value_lu:ValueLookup) -> bool:
        '''
        Combine a sub graph to a big graph.
        The return value indicate whether succeed or not.
        '''
        inputs_to_add = []
        oprs_to_add = []
        values_to_add = []
        
        # 1. add the input vertices
        matched_cnt = defaultdict(int)
        connected = False
        for in_node in graph.inputs_:
            if in_node.is_param_:
                inputs_to_add.append(in_node)
                continue

            # when the big graph is empty or the connection is done by another input node
            if len(inputs) == 0 or connected:
                inputs_to_add.append(in_node)
                values_to_add.append(in_node.value_)
                connected = True
                continue
            
            # TODO: add use penalty
            # try to match the input vertex with an existed value
            matched_v = self._sample_match_value(in_node.value_, value_lu, matched_cnt)
            if matched_v is None:
                return False
            # else ...
            
            
            matched_cnt[matched_v] += 1

            # TODO: if matched_v is not None, then connect matched v and in_node.value_

        # 2. add the operation vertices
        for op_node in graph.oprs_:
            oprs_to_add.append(op_node)
            for value in op_node.outputs_:
                values_to_add.append(value)
        
        # 3. remove output vertices
        for out_node in graph.outputs_:
            out_node.value_.uses_.remove(out_node)
        
        inputs += inputs_to_add
        oprs += oprs_to_add
        for v in values_to_add:
            value_lu.add(v)
        return True

    def _finetune_dtype(self, graph:Graph, dtype:DataType):
        # TODO: use smt to do this
        for opr in graph.oprs_:
            for value in opr.inputs_ + opr.outputs_:
                value.type_.dtype_ = dtype

    def _finetune_shape(self, graph:Graph, in_value:Value, shape:List[int]):
        in_value.type_.shape_ = shape
        value_lu = ValueLookup()

        oprs_to_fit:List[Operation] = [opr for opr in in_value.uses_ if isinstance(opr, Operation)]
        fitted_oprs = []
        value_lu.add(in_value)
        while(oprs_to_fit):
            # choose an opr to finetune
            opr_to_fit = oprs_to_fit.pop(0)
            fitted_inputs = {vid:v.type_ for vid, v in enumerate(opr_to_fit.inputs_) if v in value_lu.values}
            fitted_outputs = {vid:v.type_ for vid, v in enumerate(opr_to_fit.outputs_) if v in value_lu.values}
            self._fit_opr_shape(opr_to_fit, fitted_inputs, fitted_outputs)
            fitted_oprs.append(opr_to_fit)

            # update oprs_to_fit by finetuned_opr, update value_lu
            for value in opr_to_fit.inputs_ + opr_to_fit.outputs_:
                if value not in value_lu.values:
                    value_lu.add(value)
                    for vtx in [value.def_] + value.uses_:
                        if isinstance(vtx, Operation) and vtx not in fitted_oprs and vtx not in oprs_to_fit:
                            oprs_to_fit.append(vtx)
    
    def _fit_opr_shape(self, opr:Operation, fitted_inputs:Dict[int, TensorType], 
                       fitted_outputs:Dict[int, TensorType]):
        try:
            self._fit_opr_shape_by_smt(opr, fitted_inputs, fitted_outputs)
        except:
            self._fit_opr_shape_by_custom(opr, fitted_inputs, fitted_outputs)
        
    def _fit_opr_shape_by_smt(self, opr:Operation, fitted_inputs:Dict[int, TensorType], 
                       fitted_outputs:Dict[int, TensorType]):
        # if DEBUG:
        #     print('Solving', opr.op_.name_, fitted_inputs, fitted_outputs)

        spec = opr.op_.spec
        known_attrs = {name:attr for name, attr in opr.attrs_}
        known_input_ranks = {t_idx:opr.inputs_[t_idx].type_.rank for t_idx in range(len(opr.inputs_))}
        known_output_ranks = {t_idx:opr.outputs_[t_idx].type_.rank for t_idx in range(len(opr.outputs_))}
        solver = TypeSolverE(spec, self._rng, fitted_inputs, fitted_outputs, known_attrs, 
                             len(opr.inputs_), len(opr.outputs_), known_input_ranks, known_output_ranks)
        # try:
        solver.solve()
        # solver.solve_initial()
        # except:
        #     return False
        
        # Check nodes in value store
        store = solver.store_
        shapes_node = store.in_shapes_
        dtypes_node = store.in_dtypes_
        assert shapes_node.len_.value is not None
        assert shapes_node.len_.value == dtypes_node.len_.value
        assert shapes_node.len_.value == len(opr.inputs_)
        num = shapes_node.len_.value

        # Finetune input values
        for t_idx in range(num):
            if t_idx in fitted_inputs:
                continue

            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            dtype_node = cast(ScalarNode, dtypes_node.children_[t_idx])
            self._fit_value(opr.inputs_[t_idx], shape_node, dtype_node)
            fitted_inputs[t_idx] = opr.inputs_[t_idx].type_

        # if DEBUG:
        #     print('Solved inputs', fitted_inputs)

        # Perform complete solve
        solver = TypeSolverE(spec, self._rng, fitted_inputs, fitted_outputs, known_attrs, 
                             len(opr.inputs_), len(opr.outputs_), known_input_ranks, known_output_ranks)
        info = solver.solve()
        # if DEBUG:
        #     print('Solved', opr.op_.name_, info.attrs_, info.in_types_, info.out_types_)

        # Finetune output values
        store = solver.store_
        shapes_node = store.out_shapes_
        dtypes_node = store.out_dtypes_
        assert shapes_node.len_.value is not None
        assert shapes_node.len_.value == dtypes_node.len_.value
        assert shapes_node.len_.value == len(opr.outputs_)
        num = shapes_node.len_.value

        for t_idx in range(num):
            if t_idx in fitted_outputs:
                continue
            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            dtype_node = cast(ScalarNode, dtypes_node.children_[t_idx])
            self._fit_value(opr.outputs_[t_idx], shape_node, dtype_node)
            fitted_outputs[t_idx] = opr.outputs_[t_idx].type_
        
        assert len(fitted_inputs) == len(opr.inputs_)
        assert len(fitted_outputs) == len(opr.outputs_)

    def _fit_opr_shape_by_custom(self, opr:Operation, fitted_inputs:Dict[int, TensorType], 
                       fitted_outputs:Dict[int, TensorType]):
        op_name = opr.op_.name_
        spec = opr.op_.spec
        known_attrs = {name:attr for name, attr in opr.attrs_}
        known_input_ranks = {t_idx:opr.inputs_[t_idx].type_.rank for t_idx in range(len(opr.inputs_))}
        known_output_ranks = {t_idx:opr.outputs_[t_idx].type_.rank for t_idx in range(len(opr.outputs_))}
        self.SC.fit(op_name, spec, fitted_inputs, fitted_outputs, known_attrs, 
                    len(opr.inputs_), len(opr.outputs_), known_input_ranks, known_output_ranks)
        
        # print(op_name, fitted_inputs, fitted_outputs, known_attrs)
        solver = TypeSolverE(spec, self._rng, fitted_inputs, fitted_outputs, known_attrs, 
                             len(opr.inputs_), len(opr.outputs_), known_input_ranks, known_output_ranks)
        solver.solve()

        # Check nodes in value store
        store = solver.store_
        shapes_node = store.in_shapes_
        dtypes_node = store.in_dtypes_
        assert shapes_node.len_.value is not None
        assert shapes_node.len_.value == dtypes_node.len_.value
        assert shapes_node.len_.value == len(opr.inputs_)
        num = shapes_node.len_.value

        # Finetune input values
        for t_idx in range(num):
            # if t_idx in fitted_inputs:
            #     continue

            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            dtype_node = cast(ScalarNode, dtypes_node.children_[t_idx])
            self._fit_value(opr.inputs_[t_idx], shape_node, dtype_node)
            fitted_inputs[t_idx] = opr.inputs_[t_idx].type_

        # Perform complete solve
        solver = TypeSolverE(spec, self._rng, fitted_inputs, fitted_outputs, known_attrs, 
                             len(opr.inputs_), len(opr.outputs_), known_input_ranks, known_output_ranks)
        info = solver.solve()
        
        # Finetune attrs
        opr.attrs_ = info.attrs_

        # Finetune output values
        store = solver.store_
        shapes_node = store.out_shapes_
        dtypes_node = store.out_dtypes_
        assert shapes_node.len_.value is not None
        assert shapes_node.len_.value == dtypes_node.len_.value
        assert shapes_node.len_.value == len(opr.outputs_)
        num = shapes_node.len_.value

        for t_idx in range(num):
            if t_idx in fitted_outputs:
                continue
            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            dtype_node = cast(ScalarNode, dtypes_node.children_[t_idx])
            self._fit_value(opr.outputs_[t_idx], shape_node, dtype_node)
            fitted_outputs[t_idx] = opr.outputs_[t_idx].type_
        
        assert len(fitted_inputs) == len(opr.inputs_)
        assert len(fitted_outputs) == len(opr.outputs_)
        
    def _fit_value(self, value:Value, shape:ArrayNode, dtype:ScalarNode):
        rank_choices = int_expr_choices(shape.len_.expr, 2, max_rank + 1)
        assert value.type_.rank in rank_choices, f'rank {value.type_.rank} not in {rank_choices}'
        dtype_choices = expr_choices(dtype.expr, common_dtypes)
        assert value.type_.dtype_ in dtype_choices

        if shape.elem_defined:
            new_shape = []
            for d_idx in range(value.type_.rank):
                dim_node = cast(ScalarNode, shape.children_[d_idx])
                assert dim_node.defined
                dim_choices = int_expr_choices(dim_node.expr, 1, max_dim+1)
                if value.type_.shape_[d_idx] not in dim_choices:
                    new_shape.append(int(self._rng.choice(dim_choices)))
                else:
                    new_shape.append(value.type_.shape_[d_idx])
            value.type_.shape_ = tuple(new_shape)
                
    def _sample_connect_point(self, match_target:Value, value_lu:ValueLookup, add_cnt:Dict[Value, int]) -> Optional[Value]:
        # Sample a value from value_lu as a connection point with input value match_target.
        rank_choices = [match_target.type_.rank]
        dtype_choices = [match_target.type_.dtype_]

        matches = list(value_lu.by_choices(rank_choices, dtype_choices))
        return None if len(matches) == 0 else self._sample_value(matches, add_cnt)

    # def _match_shape_by_value(self, match_target:Value, value:Value):
    #     return True


'''
1. CombGraphGenerator: Combine patterns to graph
2. Pattern shape finetune
3. Pattern connection
'''