from typing import Iterable, List, Dict, cast, Tuple
from collections import defaultdict
from GenCoG.gencog.graph import Operation, Input
from GenCoG.gencog.graph.base import Graph, Output, Value, VertexKind, ValueType
from GenCoG.gencog.solve import TensorType

class GraphUtil:
    @staticmethod
    def copy(g:Graph):
        '''
        Deeply copy a graph, return the copied object.
        '''
        values:List[Value] = []
        for opr in g.oprs_:
            for value in opr.outputs_ + opr.inputs_:
                if value not in values:
                    values.append(value)

        # Copy values
        new_values:List[Value] = []
        value_map:Dict[Value, Value] = {}
        param_idc:Dict[Value, bool] = defaultdict(lambda :False)    # indicate if a value is a parameter
        for value in values:
            new_value = Value(TensorType(copy_iter(value.type_.shape_), value.type_.dtype_))
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
                    # assert param_idc[new_value] == True
                    param_idc[new_value] == True
        
        # Copy operators
        new_oprs = []
        for opr in g.oprs_:
            opr_inputs = [value_map[v] for v in opr.inputs_]
            opr_outputs = [value_map[v] for v in opr.outputs_]
            new_oprs.append(Operation(opr.op_, copy_attrs(opr.attrs_), opr_inputs, opr_outputs))
        
        outputs = [Output(v) for v in new_values if len(v.uses_) == 0]
        # inputs = [Input.from_value(v, param_idc[v]) for v in new_values if v.def_ is None]
        inputs = [Input.from_value(value_map[i.value_], param_idc[value_map[i.value_]]) for i in g.inputs_]
        return Graph(inputs, outputs, new_oprs)

def copy_attrs(attrs:List[Tuple[str, ValueType]]) -> List[Tuple[str, ValueType]]:
    new_attrs = []
    for attr in attrs:
        new_attrs.append((attr[0], copy_iter(attr[1])))
    return new_attrs
    
def copy_iter(attr:ValueType) -> ValueType:
    if not isinstance(attr, Iterable) or isinstance(attr, str):
        return attr
    elif isinstance(attr, List):
        return [copy_iter(e) for e in attr]
    elif isinstance(attr, Tuple):
        return tuple(copy_iter(e) for e in attr)
    else:
        raise Exception(f'Cannot copy iterable type {type(attr)}.')

