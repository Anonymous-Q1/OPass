import networkx as nx
from typing import Dict, List, cast
from numpy.random import Generator
from tvm import relay, IRModule

from GenCoG_cl.gencog.graph.base import Constant, Global, GraphVisitor, Input, Value, Output, Operation, Graph, Vertex
from GenCoG_cl.gencog.solve.solver import RelayType, RelayTypeKind, TensorType, TupleTensorType, VarTensorType
from GenCoG_cl.gencog.util import NameGenerator
from GenCoG_cl.gencog.graph.relay import build_graph

class GraphAbstractor(GraphVisitor[None]):
    def __init__(self, name: str):
        super().__init__()
        self._abs = nx.DiGraph(name=name)

        self._def_str: Dict[Value, str] = dict()
        self._inp_gen = NameGenerator('in')
        self._out_gen = NameGenerator('out')
        self._opr_gen = NameGenerator('opr')
        self._imm_gen = NameGenerator('imm')
        self._glo_gen = NameGenerator('glob')
        self._con_gen = NameGenerator('const')

    def get_graph_from_mod(self, mod:IRModule, rng:Generator):
        main_fn = mod['main']
        params = gen_tensor_value_dict(main_fn.params[1:], rng)
        graph = build_graph(mod, params)
        self.abstract(graph)
        return self._abs

    def abstract(self, graph: Graph):
        for outref in graph.outrefs_:
            self.visit(outref.def_)

        for out in graph.outputs_:
            self.visit(out)

        # Visit unvisited operators.
        for opr in graph.oprs_:
            if opr not in self._vert_memo:
                self.visit(opr)
        return self._abs

    def visit_input(self, i: Input):
        name = self._inp_gen.generate()
        self._abs.add_node(name, type='tensor', shape=tuple(i.value_.type_.shape_), dtype=str(i.value_.type_.dtype_))  # , out_order=None
        self._def_str[i.value_] = name

    def visit_output(self, o: Output):
        self.visit(o.value_.def_)
        name = self._def_str[o.value_]
        assert name in self._abs.nodes
        # self._abs.nodes[name]['in_order'] = None

        out_name = self._out_gen.generate()
        mapping = {name:out_name}
        self._abs = nx.relabel_nodes(self._abs, mapping, copy=False)
        self._def_str[o.value_] = out_name

    def visit_operation(self, opr: Operation):
        for v in opr.inputs_:
            self.visit(v.def_)
        name = self._opr_gen.generate()
        attrs = {n:v for n, v in opr.attrs_}
        self._abs.add_node(name, type='op', op=opr.op_.name_, attrs=attrs)

        for i, v in enumerate(opr.inputs_):
            in_name = self._def_str[v]
            assert in_name in self._abs.nodes
            # self._abs.nodes[in_name]['in_order'] = i
            self._abs.add_edge(in_name, name, order=i, direction='in')

        for i, v in enumerate(opr.outputs_):
            out_name = self._imm_gen.generate()
            self._abs.add_node(out_name, type='tensor', shape=tuple(v.type_.shape_), dtype=str(v.type_.dtype_))    # , out_order=i
            self._abs.add_edge(name, out_name, order=i, direction='out')
            self._def_str[v] = out_name
        
    def visit_global(self, g: Global):
        raise NotImplementedError
    
    def visit_constant(self, c: Constant):
        raise NotImplementedError

def mul_tuple(tup):
    res = 1
    for e in tup:
        res *= e
    return res

class GraphAbsForMem(GraphVisitor[None]):
    def __init__(self, name: str):
        '''
        Graph Abstractor for memory footprint simulation.
        Ignore the nodes inside an fn.
        '''
        super().__init__()
        self._abs = nx.DiGraph(name=name)

        self._def_str: Dict[Value, str] = dict()
        self._inp_gen = NameGenerator('in')
        self._out_gen = NameGenerator('out')
        self._opr_gen = NameGenerator('opr')
        self._imm_gen = NameGenerator('imm')
        self._glo_gen = NameGenerator('glob')
        self._con_gen = NameGenerator('const')

        self._visiting_fn = False
    
    def get_graph_from_mod(self, mod:IRModule, rng:Generator):
        main_fn = mod['main']
        params = gen_tensor_value_dict(main_fn.params[1:], rng)
        graph = build_graph(mod, params)
        self.abstract(graph)
        return self._abs

    def abstract(self, graph: Graph):
        for outref in graph.outrefs_:
            self.visit(outref.def_)

        for out in graph.outputs_:
            self.visit(out)

        # Visit unvisited operators.
        for opr in graph.oprs_:
            if opr not in self._vert_memo:
                self.visit(opr)
        return self._abs

    def bits_alloc(self, ty: RelayType):
        if ty.is_tensor:
            if ty.kind == RelayTypeKind.tensor:
                ty = cast(TensorType, ty)
                return ty.dtype_.bits_ * mul_tuple(ty.shape_)
            elif ty.kind == RelayTypeKind.tuple:
                ty = cast(TupleTensorType, ty)
                mem = 0
                for elem in ty.elems_:
                    mem += self.bits_alloc(elem)
                return mem
            else:
                ty = cast(VarTensorType, ty)
                assert ty.is_instantiated
                return self.bits_alloc(ty.type_)
        else:
            return 0

    def visit(self, v: Vertex):
        if self._visiting_fn:
            self._vert_memo[v] = None
            if isinstance(v, Operation):
                for inp in v.inputs_:
                    self.visit(inp.def_)
                return
            elif isinstance(v, Constant):
                return
            elif isinstance(v, Global):
                return
            else:
                raise Exception(type(v))
        else:
            return super().visit(v)

    def visit_input(self, i: Input):
        name = self._inp_gen.generate()
        self.add_tensor_node(name, i.value_)
        self._def_str[i.value_] = name

    def visit_output(self, o: Output):
        self.visit(o.value_.def_)
        name = self._def_str[o.value_]
        assert name in self._abs.nodes
        # self._abs.nodes[name]['in_order'] = None

        out_name = self._out_gen.generate()
        mapping = {name:out_name}
        self._abs = nx.relabel_nodes(self._abs, mapping, copy=False)
        self._def_str[o.value_] = out_name

    def visit_operation(self, opr: Operation):
        inputs = opr.inputs_
        if opr.op_.name_ == 'call':
            self._visiting_fn = True
            self.visit(opr.inputs_[0].def_)
            inputs = opr.inputs_[1:]
            self._visiting_fn = False

        for v in inputs:
            self.visit(v.def_)
        name = self._opr_gen.generate()
        attrs = {n:v for n, v in opr.attrs_}
        self._abs.add_node(name, type='op', op=opr.op_.name_, attrs=attrs)
                        #    shape=([o.type_ for o in opr.inputs_], [o.type_ for o in opr.outputs_]))

        for i, v in enumerate(inputs):
            in_name = self._def_str[v]
            assert in_name in self._abs.nodes
            # self._abs.nodes[in_name]['in_order'] = i
            self._abs.add_edge(in_name, name, order=i, direction='in')

        for i, v in enumerate(opr.outputs_):
            out_name = self._imm_gen.generate()
            
            self.add_tensor_node(out_name, v)
            self._abs.add_edge(name, out_name, order=i, direction='out')
            self._def_str[v] = out_name    

    def visit_global(self, g: Global):
        name = self._glo_gen.generate()
        self.add_tensor_node(name, g.value_)
        self._def_str[g.value_] = name
    
    def visit_constant(self, c: Constant):
        name = self._con_gen.generate()
        self.add_tensor_node(name, c.value_)
        self._def_str[c.value_] = name

    def add_tensor_node(self, name: str, value: Value):
        self._abs.add_node(name, type='tensor', mem=self.bits_alloc(value.type_))

class GraphVisitorForGen(GraphVisitor[None]):
    def __init__(self):
        '''
        Graph Abstractor for combinational grpah generation.
        Find all available value for generation, exluding values inside an fn.
        '''
        super().__init__()
        self.available_values: List[Value] = []
        self._visiting_fn = False

    def abstract_values(self, graph: Graph):
        for outref in graph.outrefs_:
            self.visit(outref.def_)

        for out in graph.outputs_:
            self.visit(out)

        # Visit unvisited operators.
        for opr in graph.oprs_:
            if opr not in self._vert_memo:
                self.visit(opr)

        for inp in graph.inputs_:
            if inp not in self._vert_memo:
                self.visit(inp)
        return self.available_values
    
    def visit(self, v: Vertex):
        if self._visiting_fn:
            self._vert_memo[v] = None
            if isinstance(v, Operation):
                for inp in v.inputs_:
                    self.visit(inp.def_)
                return
            elif isinstance(v, Constant):
                return
            elif isinstance(v, Global):
                return
            else:
                raise Exception(type(v))
        else:
            return super().visit(v)

    def visit_input(self, i: Input):
        if i.value_.type_.kind == RelayTypeKind.tensor and i.value_ not in self.available_values:
            self.available_values.append(i.value_)
    
    def visit_output(self, o: Output):
        self.visit(o.value_.def_)
    
    def visit_operation(self, opr: Operation):
        inputs = opr.inputs_
        if opr.op_.name_ == 'call':
            self._visiting_fn = True
            self.visit(opr.inputs_[0].def_)
            inputs = opr.inputs_[1:]
            self._visiting_fn = False

        for v in inputs:
            self.visit(v.def_)

        for o in opr.outputs_:
            if o.type_.kind == RelayTypeKind.tensor and o not in self.available_values:
                self.available_values.append(o)
            
    def visit_global(self, g: Global):
        if g.value_.type_.kind == RelayTypeKind.tensor and g.value_ not in self.available_values:
            self.available_values.append(g.value_)
    
    def visit_constant(self, c: Constant):
        if c.value_.type_.kind == RelayTypeKind.tensor and c.value_ not in self.available_values:
            self.available_values.append(c.value_)

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)

def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}