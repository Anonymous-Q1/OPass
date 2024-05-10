from numpy.random import Generator
from typing import List, cast, Iterable
from tvm.relay import parse

from . import Pattern
from .rule import ReshapeRule
from ..util import load_gmod_from_file
from ..sugar import Temp
from GenCoG_cl.gencog.graph import Graph, Input, GraphMod, print_relay, build_graph, Operation
from GenCoG_cl.gencog.solve.solver import TensorType, RelayTypeKind, TupleTensorType, RelayType

max_dim = 100
min_dim = 1

class ReshapeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def reshaper(pattern: Pattern, target_shape: List[int], rng: Generator) -> GraphMod:
    return PatternReshaper(pattern, rng).reshape(target_shape)

class PatternReshaper:
    def __init__(self, pattern: Pattern, rng: Generator) -> None:
        self._rng = rng
        self.pattern_ = pattern

    def reshape(self, target_shape: List[int]) -> GraphMod:
        # Copy the graph module to avoid pollution.
        gmod = load_gmod_from_file(self.pattern_.path_)

        if self.pattern_.rule_ is not None:
            rule = self.pattern_.rule_.reshapeR_
            self.reshape_anchor_in(target_shape, gmod, rule)
            self.reshape_rel_ins(target_shape, gmod, rule)
            self.reshape_oprs(target_shape, gmod, rule)

        # Use relay to automatically refer type.
        new_code = print_relay(gmod)
        new_gmod = build_graph(parse(new_code))

        # Validate the new gmod
        self.validate(new_gmod)

        return new_gmod
    
    def reshape_anchor_in(self, target_shape: List[int], gmod: GraphMod, rule: ReshapeRule):
        # Choose the anchor input from rule
        anchor_idx = rule.anchor_idx
        anchor_input = gmod['main'].inputs_[anchor_idx]

        # Reshape the anchor input according to target_shape with rule check
        anchor_inR = rule.inRs_[anchor_idx]
        # rank check
        if len(target_shape) < anchor_inR.rank_range_[0] or len(target_shape) > anchor_inR.rank_range_[1]:
            raise ReshapeError(f'Target shape rank {len(target_shape)} does not conform rule rank range {anchor_inR.rank_range_}.')
        # shape check
        for relR in anchor_inR.rel_anchor_:
            if eval(relR)(target_shape) != True:
                raise ReshapeError(f'Target shape {target_shape} does not confor rule {relR}.')
        # reshape
        anchor_input.value_.type_.shape_ = tuple(target_shape)
        return anchor_idx

    def reshape_rel_ins(self, target_shape: List[int], gmod: GraphMod, rule: ReshapeRule):
        # Reshape other inputs
        anchor_idx = rule.anchor_idx
        for idx, inp in enumerate(gmod['main'].inputs_):
            if idx == anchor_idx:
                continue
            inR = rule.inRs_[idx]
            new_shape = []
            for _, relR in enumerate(inR.rel_anchor_):
                try:
                    dim = eval(relR)(target_shape)
                except:
                    dim = None

                if dim is None:
                    continue
                if dim == 'any':
                    dim = self._rng.integers(min_dim, max_dim+1)
                new_shape.append(int(dim))
            # reshape
            inp.value_.type_.shape_ = tuple(new_shape)

    def reshape_oprs(self, target_shape: List[int], gmod: GraphMod, rule: ReshapeRule):
        # Reshape neccesary oprs
        for oprR in rule.oprRs_:
            # Locate the opr vertex in Graph
            opr = self.locate_opr(gmod['main'], oprR.op_, oprR.pos_)

            # Original attrs
            opr_attrs = {}
            opr_attrs.update(opr.attrs_)

            # Update attrs
            for attr_name in oprR.attrs_:
                rel_anchor = oprR[attr_name]
                
                old_attr = opr_attrs[attr_name]
                new_attr = []
                for _, relR in enumerate(rel_anchor):
                    try:
                        dim = eval(relR)(target_shape)
                    except:
                        dim = None

                    if dim is None:
                        continue
                    if dim == 'any':
                        dim = self._rng.integers(min_dim, max_dim+1)
                    new_attr.append(int(dim))
                
                if len(new_attr) == 1 and not isinstance(old_attr, Iterable):
                    new_attr = new_attr[0]
                
                # update
                opr_attrs[attr_name] = new_attr
            opr.attrs_ = list(opr_attrs.items())

    
    def locate_opr(self, graph: Graph, op: str, pos: str):
        start = graph.inputs_[int(pos[0])].value_
        cursor = start.uses_[int(pos[1])]
        cursor = cast(Operation, cursor)
        for direction in pos[2:]:
            assert len(cursor.outputs_) == 1
            opr_out = cursor.outputs_[0]
            cursor = opr_out.uses_[int(direction)]
        
        assert cursor.op_.name_ == op, f'{cursor.op_.name_}!={op}'
        return cursor
            
    def validate(self, gmod: GraphMod):
        '''
        Check whether the gmod is valid.
        '''

        # If it contains negative shape.
        graph = gmod['main']
        for opr in graph.oprs_:
            for out in opr.outputs_:
                self.check_negative_type(out.type_)
    
    def check_negative_type(self, ty: RelayType):
        if isinstance(ty, TensorType):
            for dim in ty.shape_:
                try:
                    d = int(dim)
                except:
                    continue
                if d < 0:
                    raise ReshapeError('Found negative shape.')
        elif isinstance(ty, TupleTensorType):
            for fty in ty.elems_:
                self.check_negative_type(fty)
