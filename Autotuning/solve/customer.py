from numpy.random import Generator
from functools import reduce
from typing import Dict, Optional
from GenCoG.gencog.solve.solver import TensorType, OpTypeInfo
from GenCoG.gencog.graph.base import ValueType
from GenCoG.gencog.spec import TypeSpec
from ..config import PARAMS

max_dim = PARAMS['max_dim']

class ShapeCustomer:
    def __init__(self, rng:Generator) -> None:
        self.rng_ = rng
        
        self.op_fit_func = {
            # 'transpose': self.fit_transpose,
            'nn.layer_norm': self.fit_nnLayerNorm,
            # 'split': self.fit_split,
            'nn.conv1d_transpose': self.fit_nnConv1dT,
            'reshape': self.fit_reshape,
        }

    def fit(self, 
            op_name: str,
            spec: TypeSpec,
            known_inputs: Dict[int, TensorType],
            known_outputs: Dict[int, TensorType], 
            known_attrs: Dict[str, ValueType], 
            known_input_num: Optional[int] = None,
            known_output_num: Optional[int] = None,
            known_input_ranks: Optional[Dict[int, int]] = {}, 
            known_output_ranks: Optional[Dict[int, int]] = {},
            ):
        self.op_fit_func[op_name](spec, known_inputs, known_outputs, known_attrs, 
                                    known_input_num, known_output_num, known_input_ranks, known_output_ranks)

    def fit_transpose(self, 
                        spec: TypeSpec,
                        known_inputs: Dict[int, TensorType],
                        known_outputs: Dict[int, TensorType], 
                        known_attrs: Dict[str, ValueType], 
                        known_input_num: Optional[int] = None,
                        known_output_num: Optional[int] = None,
                        known_input_ranks: Optional[Dict[int, int]] = {}, 
                        known_output_ranks: Optional[Dict[int, int]] = {},
                        ):
        known_input_num == _assert_value(known_input_num, 1)
        known_output_num == _assert_value(known_output_num, 1)
        attr_axes = known_attrs['axes']
        
        # print('in', known_inputs)
        # print('out', known_outputs)
        # print('attr', known_attrs)

        # Calculate outputs from inputs
        if len(known_inputs) != 0:
            assert len(known_inputs) == known_input_num
            assert len(known_outputs) == 0
            in_shape = known_inputs[0].shape_
            assert len(attr_axes) == len(in_shape)
            out_shape = []
            for axis in attr_axes:
                out_shape.append(in_shape[axis])
            known_outputs[0] = TensorType(out_shape, known_inputs[0].dtype_)
            return
            
        # Calculate inputs from outputs
        if len(known_outputs) != 0:
            assert len(known_outputs) == known_output_num
            assert len(known_inputs) == 0
            out_shape = known_outputs[0].shape_
            assert len(attr_axes) == len(out_shape)
            in_shape = [0] * len(attr_axes)
            for i, axis in enumerate(attr_axes):
                in_shape[axis] = out_shape[i]
            known_inputs[0] = TensorType(in_shape, known_outputs[0].dtype_)
            return
    
    def fit_nnLayerNorm(self, 
                        spec: TypeSpec,
                        known_inputs: Dict[int, TensorType],
                        known_outputs: Dict[int, TensorType], 
                        known_attrs: Dict[str, ValueType], 
                        known_input_num: Optional[int] = None,
                        known_output_num: Optional[int] = None,
                        known_input_ranks: Optional[Dict[int, int]] = {}, 
                        known_output_ranks: Optional[Dict[int, int]] = {},
                        ):
        known_input_num == _assert_value(known_input_num, 3)
        known_output_num == _assert_value(known_output_num, 1)
        attr_axis = known_attrs['axis']
        
        # print('in', known_inputs)
        # print('out', known_outputs)
        # print('attr', known_attrs)

        if 0 not in known_inputs and (1 in known_inputs or 2 in known_inputs):
            target_shape = known_inputs[1] if 1 in known_inputs else known_inputs[2]
            assert 0 in known_input_ranks
            in_shape = [int(self.rng_.integers(1, max_dim+1)) for _ in range(known_input_ranks[0])]
            in_shape[attr_axis] = target_shape.shape_[0]
            known_inputs[0] = TensorType(in_shape, target_shape.dtype_)

    def fit_nnConv1dT(self, 
                    spec: TypeSpec,
                    known_inputs: Dict[int, TensorType],
                    known_outputs: Dict[int, TensorType], 
                    known_attrs: Dict[str, ValueType], 
                    known_input_num: Optional[int] = None,
                    known_output_num: Optional[int] = None,
                    known_input_ranks: Optional[Dict[int, int]] = {}, 
                    known_output_ranks: Optional[Dict[int, int]] = {},
                    ):
        
        # print('in', known_inputs)
        # print('out', known_outputs)
        # print('attr', known_attrs)
        for k in list(known_attrs.keys()):
            known_attrs.pop(k)

    def fit_reshape(self, 
                    spec: TypeSpec,
                    known_inputs: Dict[int, TensorType],
                    known_outputs: Dict[int, TensorType], 
                    known_attrs: Dict[str, ValueType], 
                    known_input_num: Optional[int] = None,
                    known_output_num: Optional[int] = None,
                    known_input_ranks: Optional[Dict[int, int]] = {}, 
                    known_output_ranks: Optional[Dict[int, int]] = {},
                    ):
        # print('in', known_inputs)
        # print('out', known_outputs)
        # print('attr', known_attrs)

        if 0 in known_inputs and 0 not in known_outputs:
            attr_shape = known_attrs['newshape']
            attr_shape_mul = reduce(lambda x,y:x*y, attr_shape, 1)
            in_shape = known_inputs[0].shape_
            in_shape_mul = reduce(lambda x,y:x*y, in_shape, 1)
            if in_shape_mul % attr_shape_mul == 0:
                fitted_attr_shape = tuple([d * int(in_shape_mul/attr_shape_mul) if i==0 else d for i,d in enumerate(attr_shape)])
                known_attrs['newshape'] = fitted_attr_shape
            elif len(in_shape) == len(attr_shape):
                fitted_attr_shape = tuple([d for d in in_shape])
                known_attrs['newshape'] = fitted_attr_shape
            elif len(in_shape) < len(attr_shape):
                fitted_attr_shape = tuple([in_shape[i] if i < len(in_shape) else 1 for i in range(len(attr_shape))])
                known_attrs['newshape'] = fitted_attr_shape
            else:
                last_attr_dim = reduce(lambda x,y:x*y, in_shape[len(attr_shape)-1:], 1)
                fitted_attr_shape = tuple([in_shape[i] if i != len(attr_shape)-1 else last_attr_dim for i in range(len(attr_shape))])
                known_attrs['newshape'] = fitted_attr_shape
            assert reduce(lambda x,y:x*y, known_attrs['newshape'], 1) == in_shape_mul
                # known_attrs.pop('newshape')
        
        if 0 in known_outputs and 0 not in known_inputs:
            out_shape = known_outputs[0].shape_
            out_shape_mul = reduce(lambda x,y:x*y, out_shape, 1)
            known_attrs['newshape'] = tuple([d for d in out_shape])
            assert 0 in known_input_ranks
            in_rank = known_input_ranks[0]
            if in_rank == len(out_shape):
                in_shape = [d for d in out_shape]
                known_inputs[0] = in_shape
            elif in_rank > len(out_shape):
                in_shape = [out_shape[i] if i < len(out_shape) else 1 for i in range(in_rank)]
                known_inputs[0] = in_shape
            else:
                last_in_dim = reduce(lambda x,y:x*y, out_shape[in_rank-1:], 1)
                in_shape = tuple([out_shape[i] if i != in_rank-1 else last_in_dim for i in range(in_rank)])
                known_inputs[0] = in_shape
            assert reduce(lambda x,y:x*y, known_inputs[0], 1) == out_shape_mul


def _assert_value(obj, value):
    if obj is None:
        return value
    else:
        assert obj == value