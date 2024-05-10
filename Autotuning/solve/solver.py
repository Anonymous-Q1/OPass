from typing import Dict, Optional, cast, Iterable
from numpy.random import Generator
from GenCoG.gencog.solve import TypeSolver, SolveError
from GenCoG.gencog.solve.solver import TensorType
from GenCoG.gencog.solve.store import ValueStore, StoreNode, ScalarNode, ArrayNode, NodeKind, ValueStatus, StoreError
from GenCoG.gencog.graph.base import ValueType
from GenCoG.gencog.spec import TypeSpec
from GenCoG.gencog.expr.visitor import CopyExpr
from GenCoG.gencog.expr.array import Tuple
from GenCoG.gencog.expr.basic import to_expr, Cmp, CmpOp, Const, ExprKind, And, Var
from GenCoG.gencog.expr.ty import _type_convert_funcs
from GenCoG.gencog.util import Ref
from GenCoG.gencog.solve.smt import solve_smt
from GenCoG.gencog.solve.valid import validate

class TypeSolverE(TypeSolver):
    def __init__(self, 
                 spec: TypeSpec, 
                 rng: Generator, 
                 known_inputs: Dict[int, TensorType],
                 known_outputs: Dict[int, TensorType], 
                 known_attrs: Dict[str, ValueType], 
                 known_input_num: Optional[int] = None,
                 known_output_num: Optional[int] = None,
                 known_input_ranks: Optional[Dict[int, int]] = {}, 
                 known_output_ranks: Optional[Dict[int, int]] = {},
                 ):
        super().__init__(spec, known_inputs, rng)
        self._known_out = known_outputs
        self._known_attr = known_attrs
        self._known_in_num = known_input_num
        self._known_out_num = known_output_num
        self._known_in_rank = known_input_ranks
        self._known_out_rank = known_output_ranks

        # Add _known_attr and _known_out to self._extra
        
        try:
            self._set_known_inputs()
            self._set_known_outputs()

            if len(known_attrs) > 0:
                self._set_known_attr()
        except:
            if len(known_attrs) > 0:
                self._set_known_attr()

            self._set_known_inputs()
            self._set_known_outputs()

    def _set_known_outputs(self):
        cp = CopyExpr()
        if self._known_out_num is not None:
            out_num = cp.copy(self._spec.out_num)

            out_shape_root = self.store_.out_shapes_
            out_shape_root.set_len_defined(out_num)
            out_shape_root.set_expr_defined(cp.copy(self._spec.out_shapes))
            self._add_extra(out_shape_root.len_, self._known_out_num)
            out_shape_root.set_len_solved(self._known_out_num)

            out_dtype_root = self.store_.out_dtypes_
            out_dtype_root.set_len_defined(out_num)
            out_dtype_root.set_expr_defined(cp.copy(self._spec.out_dtypes))
            out_dtype_root.set_len_solved(self._known_out_num)

            dtypes = self._partial.transform(out_dtype_root.expr_)
            assert dtypes.kind == ExprKind.TUPLE
            out_dtype_root.set_elem_defined(dtypes)
        else:
            return
        
        # Set known ranks
        ranks = self._partial.transform(cp.copy(self._spec.out_ranks))
        assert ranks.kind == ExprKind.TUPLE
        ranks = cast(Tuple, ranks)
        assert out_shape_root.len_.value_ == len(ranks.fields_), f'{out_shape_root.len_.value_}!={len(ranks.fields_)}'
        
        for t_idx, tensor in enumerate(out_shape_root.children_):
            tensor = cast(ArrayNode, tensor)
            tensor.set_len_defined(ranks.fields_[t_idx])
            if t_idx in self._known_out:
                known_rank = self._known_out[t_idx].rank
                self._add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)
            elif t_idx in self._known_out_rank:
                known_rank = self._known_out_rank[t_idx]
                self._add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)

        shapes = self._partial.transform(out_shape_root.expr_)
        assert shapes.kind == ExprKind.TUPLE
        out_shape_root.set_elem_defined(shapes)

        for t_idx in self._known_out:
            shape_node = cast(ArrayNode, out_shape_root.children_[t_idx])
            shape = self._partial.transform(shape_node.expr_)
            assert shape.kind == ExprKind.TUPLE
            shape_node.set_elem_defined(shape)
            
            for d_idx, dim in enumerate(self._known_out[t_idx].shape_):
                dim_node = cast(ScalarNode, shape_node.children_[d_idx])
                self._add_extra(dim_node, dim)
                dim_node.set_solved(dim)

        # Set known dtype
        dtypes = self._partial.transform(out_dtype_root.expr_)
        assert dtypes.kind == ExprKind.TUPLE
        dtypes = cast(Tuple, dtypes)
        out_dtype_root.set_elem_defined(dtypes)

        for t_idx, dtype in enumerate(out_dtype_root.children_):
            dtype = cast(ScalarNode, dtype)
            if t_idx in self._known_out:
                known_dtype = self._known_out[t_idx].dtype_
                self._add_extra(dtype, known_dtype)
                dtype.set_solved(known_dtype)
        
    def _set_known_attr(self):
        for a_name, attr_node in self.store_.attrs_:
            if a_name in self._known_attr:
                self._set_solved_node(attr_node, self._known_attr[a_name])

    def _set_known_inputs(self):
        if self._known_in_num is not None:
            shape_root = self.store_.in_shapes_
            shape_root.set_len_solved(self._known_in_num)
            dtype_root = self.store_.in_dtypes_
            dtype_root.set_len_solved(self._known_in_num)
        else:
            return
        
        # Set known rank
        ranks = self._partial.transform(self._in_ranks)
        assert ranks.kind == ExprKind.TUPLE
        ranks = cast(Tuple, ranks)
        assert shape_root.len_.value_ == len(ranks.fields_), f'{shape_root.len_.value_}!={len(ranks.fields_)}'
        
        for t_idx, tensor in enumerate(shape_root.children_):
            tensor = cast(ArrayNode, tensor)
            tensor.set_len_defined(ranks.fields_[t_idx])
            if t_idx in self._known:
                known_rank = self._known[t_idx].rank
                self._add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)
            elif t_idx in self._known_in_rank:
                known_rank = self._known_in_rank[t_idx]
                self._add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)

        # Set known shape
        shapes = self._partial.transform(shape_root.expr_)
        assert shapes.kind == ExprKind.TUPLE
        shape_root.set_elem_defined(shapes)

        for t_idx, tensor in enumerate(shape_root.children_):
            if t_idx in self._known:
                tensor = cast(ArrayNode, tensor)

                shape = self._partial.transform(tensor.expr_)
                assert shape.kind == ExprKind.TUPLE
                tensor.set_elem_defined(shape)
                
                for d_idx, dim in enumerate(tensor.children_):
                    dim = cast(ScalarNode, dim)
                    known_dim = self._known[t_idx].shape_[d_idx]
                    self._add_extra(dim, known_dim)
                    dim.set_solved(known_dim)
        
        # Set known dtype
        dtypes = self._partial.transform(dtype_root.expr_)
        assert dtypes.kind == ExprKind.TUPLE
        dtypes = cast(Tuple, dtypes)
        dtype_root.set_elem_defined(dtypes)

        for t_idx, dtype in enumerate(dtype_root.children_):
            dtype = cast(ScalarNode, dtype)
            if t_idx in self._known:
                known_dtype = self._known[t_idx].dtype_
                self._add_extra(dtype, known_dtype)
                dtype.set_solved(known_dtype)


    def _solve_one_iter(self):
        # Solve attributes
        changed = False
        try:
            for _, node in self.store_.attrs_:
                changed |= self._solve_node(node)

            # Solve inputs
            changed |= self._solve_shapes(self.store_.in_shapes_)
            changed |= self._solve_dtypes(self.store_.in_dtypes_)
            # Solve extra constraints
            changed |= self._solve_extra()
        except StoreError as err:
            raise SolveError(self, err.msg_)

        return changed
    
    def _solve_smt(self):
        # Find all valid variables and constraints with union-find
        changed = False
        union = validate(self.store_, self._extra)
        all_valid = list(union.all_valid())
        if len(all_valid) == 0:
            return False

        # Process valid expressions
        all_vars = []
        extra = []
        for e in all_valid:
            # Differentiate variables and constraints
            if e.kind != ExprKind.VAR:
                extra.append(e)
                continue
            var = cast(Var, e)
            if Ref(var) in all_vars:
                continue

            # Try sample variables only bounded by its own range
            if union.has_use(var):
                all_vars.append(Ref(var))
            elif self._try_sample(var):
                changed = True
            else:
                raise SolveError(
                    self, 'Cannot solve unconstrained variable.'
                )

        # Solve by SMT
        changed |= solve_smt(all_vars, extra, self.store_, self._rng)
        return changed
    
    def _solve_shapes(self, root: ArrayNode) -> bool:
        # Solve number
        changed = False
        if not root.len_solved:
            changed |= self._solve_len(root, by_elem=False)
            if not root.len_solved:
                return changed

        # Solve tensors
        if not root.elem_defined:
            # Partially evaluate ranks
            ranks = self._partial.transform(self._in_ranks)
            if ranks.kind != ExprKind.TUPLE:
                return changed
            ranks = cast(Tuple, ranks)
            if root.len_.value != len(ranks.fields_):
                raise SolveError(
                    self,
                    f'Length of input rank array {len(ranks.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define ranks for each input tensor
            for tensor, rank in zip(root.children_, ranks.fields_):
                tensor = cast(ArrayNode, tensor)
                tensor.set_len_defined(rank)

            # Partially evaluate shapes
            shapes = self._partial.transform(root.expr_)
            if shapes.kind != ExprKind.TUPLE:
                return changed
            shapes = cast(Tuple, shapes)
            if root.len_.value != len(shapes.fields_):
                raise SolveError(
                    self,
                    f'Length of input shape array {len(shapes.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define shapes for each input tensor
            root.set_elem_defined(shapes)
            changed = True

        # Solve shapes
        for t_idx, tensor in enumerate(root.children_):
            # Solve rank
            tensor = cast(ArrayNode, tensor)
            prev_solved = tensor.len_solved
            changed |= self._solve_len(tensor, by_elem=False)
            if t_idx in self._known:
                known_rank = self._known[t_idx].rank
                self._try_add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)
            elif t_idx in self._known_in_rank:
                known_rank = self._known_in_rank[t_idx]
                self._try_add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)
            changed |= prev_solved != tensor.len_solved
            if not tensor.len_solved:
                continue

            # Partially evaluate dimensions
            if not tensor.elem_defined:
                shape = self._partial.transform(tensor.expr_)
                if shape.kind != ExprKind.TUPLE:
                    continue
                shape = cast(Tuple, shape)
                if tensor.len_.value != len(shape.fields_):
                    raise SolveError(
                        self,
                        f'Length of input shape {len(shape.fields_)} for tensor {t_idx} is not '
                        f'consistent with rank {tensor.len_.value}. '
                    )
                tensor.set_elem_defined(shape)
                changed = True

            # Solve dimensions
            for d_idx, dim in enumerate(tensor.children_):
                dim = cast(ScalarNode, dim)
                prev_solved = dim.solved
                changed |= self._solve_scalar(dim)
                if t_idx in self._known:
                    known_dim = self._known[t_idx].shape_[d_idx]
                    self._try_add_extra(dim, known_dim)
                    dim.set_solved(known_dim)
                changed |= prev_solved != dim.solved

        return changed

    def _solve_extra(self) -> bool:
        new_extra = []
        changed = False

        for e in self._extra:
            post = self._partial.transform(e)
            if post.kind == ExprKind.CONST:
                const = cast(Const, post)
                if const.val_ is False:
                    # print(e)
                    raise SolveError(
                        self, 'Extra constraint is not satisfiable.'
                    )
                changed = True
                continue
            elif post.kind == ExprKind.AND:
                and_e = cast(And, post)
                new_extra.extend(and_e.clauses_)
                changed = True
                continue
            elif post.kind == ExprKind.CMP:
                cmp = cast(Cmp, post)
                if cmp.op_ == CmpOp.EQ and cmp.lhs_.kind == ExprKind.VAR and \
                        cmp.rhs_.kind == ExprKind.CONST:
                    lhs = cast(Var, cmp.lhs_)
                    rhs = cast(Const, cmp.rhs_)
                    self.store_.set_var_solved(lhs, rhs.val_)
                    changed = True
                    continue
        
            changed |= not self._eq.visit(e, post)
            new_extra.append(post)

        self._extra = new_extra
        return changed

    def _set_solved_node(self, node:StoreNode, val:ValueType):
        if node.kind == NodeKind.SCALAR:
            self._set_solved_scalar(cast(ScalarNode, node), val)
        else:
            self._set_solved_array(cast(ArrayNode, node), val)
    
    def _set_solved_scalar(self, node:ScalarNode, val:ValueType):
        assert node.status_ != ValueStatus.UNDEFINED
        if node.type_ in _type_convert_funcs:
            val = _type_convert_funcs[node.type_](val)
        node.set_solved(val)
        # self._add_extra(node, val)

    def _set_solved_array(self, node:ArrayNode, val:ValueType):
        assert isinstance(val, Iterable)
        if not node.len_solved:
            node.set_len_solved(len(val))
            # self._add_extra(node.len_, len(val))
        
        if node.expr_defined and not node.elem_defined:
            tup = self._partial.transform(node.expr_)
            assert tup.kind == ExprKind.TUPLE or tup.kind == ExprKind.LIST
            node.set_elem_defined(tup)
        
        for child, cv in zip(node.children_, val):
            self._set_solved_node(child, cv)

    
    def _add_extra(self, node:ScalarNode, val:ValueType):
        if node.expr_ is None:
            return
        self._extra.append(Cmp(CmpOp.EQ, node.expr_, Const(val)))