from typing import List, Dict, Tuple, cast, isinstance
from GenCoG_cl.gencog.graph import Graph, GraphMod, GraphVisitor
from GenCoG_cl.gencog.graph.base import Constant, Global, Input, Operation, Output, Value, Vertex
from GenCoG_cl.gencog.solve import RelayType, RelayTypeKind, TensorType
from GenCoG_cl.gencog.graph.relay import virtual_ops
from Autotuning.solve import TypeSolverE

class GraphReshaper(GraphVisitor[None]):
    def __init__(self, gmod: GraphMod) -> None:
        super().__init__()
        self.gmod_ = gmod
        self._value_types: Dict[Value, RelayType] = {}

    def reshape(self, start_input: Input, target_type: TensorType):
        """
        Reshape the graph by reshape the start_input's shape to target_shape,
        along with fintune other values' shape and operators' attributes.
        """
        self.visit_value(start_input.value_, target_type)
        self.visit(start_input)
        return self.gmod_

    def visit(self, v: Vertex):
        # Avoid recursive visiting.
        if v in self._vert_memo:
            return 
        self._vert_memo[v] = None
        self._methods[v.kind](v)

    def visit_value(self, v: Value, ty: RelayType):
        v.type_ = ty
        self._value_types[v] = ty
    
    def visit_input(self, i: Input):
        assert i.value_ in self._value_types
        for vtx in i.value_.uses_:
            self.visit(vtx)

    def visit_operation(self, opr: Operation) -> None:
        reshaped_ins = [v for v in opr.inputs_ if v in self._value_types]
        reshaped_outs = [v for v in opr.outputs_ if v in self._value_types]
        
        # Reshape the operation vertex as well as its input and output values.
        if opr.op_.name_ not in virtual_ops:
            self._reshape_opr(opr, reshaped_ins, reshaped_outs)
        
        else:
            raise Exception(f'Unsupported op name: {opr.op_.name_}.')
        
        # Visit connected vertices.
        for v in opr.inputs_ + opr.outputs_:
            self.visit(v.def_)
            for vtx in v.uses_:
                self.visit(vtx)

    def visit_output(self, o: Output) -> None:
        return
    
    def visit_constant(self, c: Constant) -> None:
        raise NotImplementedError 
    
    def visit_global(self, g: Global) -> None:
        raise NotImplementedError 
        
    def _reshape_normal_opr(self, opr: Operation, reshaped_ins: List[Value], reshaped_outs: List[Value]):
        self._reshape_by_smt(opr, reshaped_ins, reshaped_outs)

    def _fit_opr_shape_by_smt(self, opr:Operation, re_in_tys:Dict[int, TensorType], 
                       re_out_tys:Dict[int, TensorType]):
        spec = opr.op_.spec
        known_attrs = {name:attr for name, attr in opr.attrs_}
        known_input_ranks = {t_idx:opr.inputs_[t_idx].type_.rank for t_idx in range(len(opr.inputs_))}
        known_output_ranks = {t_idx:opr.outputs_[t_idx].type_.rank for t_idx in range(len(opr.outputs_))}
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
            if t_idx in fitted_inputs:
                continue

            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            dtype_node = cast(ScalarNode, dtypes_node.children_[t_idx])
            self._fit_value(opr.inputs_[t_idx], shape_node, dtype_node)
            fitted_inputs[t_idx] = opr.inputs_[t_idx].type_


        # Perform complete solve
        solver = TypeSolverE(spec, self._rng, fitted_inputs, fitted_outputs, known_attrs, 
                             len(opr.inputs_), len(opr.outputs_), known_input_ranks, known_output_ranks)

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
        
    


    
'''
    def reshape(self, start_input: Input, target_shape: List[int]):
        """
        Reshape the graph by reshape the start_input's shape to target_shape,
        along with fintune other values' shape and operators' attributes.
        """
        start_input.value_.type_.shape_ = target_shape
        reshaped_values = []
        reshaped_values.append(start_input.value_)
        reshaped_oprs = []

        oprs_to_re: List[Operation] = [opr for opr in start_input.value_.uses_ if isinstance(opr, Operation)]
        while(oprs_to_re):
            # Choose an opr to reshape
            opr_to_re = oprs_to_re.pop(0)
            reshaped_inputs = [v for v in opr_to_re.inputs_ if v in reshaped_values]
            reshaped_outputs = [v for v in opr_to_re.outputs_ if v in reshaped_values]
            self.reshape_opr(opr_to_re, reshaped_inputs, reshaped_outputs)
            
            # Update oprs_to_re, reshaped_values and reshaped_oprs
            reshaped_oprs.append(opr_to_re)
            for value in opr_to_re.inputs_ + opr_to_re.outputs_:
                if value not in reshaped_values:
                    reshaped_values.append(value)
                    for vtx in [value.def_] + value.uses_:
                        if isinstance(vtx, Operation):
                            if vtx not in reshaped_oprs and vtx not in oprs_to_re:
                                oprs_to_re.append(vtx)
                        else:
                            raise Exception('Cannot handel vtx with type', type(vtx))
'''