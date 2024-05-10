from typing import Iterable, List
from numpy.random import Generator
from GenCoG.gencog.graph import GraphGenerator, Operation, Input
from GenCoG.gencog.spec import Op
from GenCoG.gencog.graph.lookup import ValueLookup
from GenCoG.gencog.config import params
from GenCoG.gencog.graph.base import Graph, Output

max_opr_num: int = params['graph.max_opr_num']
opr_trials: int = params['graph.opr_trials']
use_penal: float = params['graph.use_penal']

class IncreGenStatus:
    def __init__(self) -> None: # inputs:List = [], oprs:List = [], value_lu:ValueLookup = ValueLookup()
        self.inputs:List[Input] = []
        self.oprs:List[Operation] = []
        self.value_lu = ValueLookup()
        self.outputs:List[Output] = []

        self.last_opr:Operation = None

        self.need_init:bool = True
    
    def clean_output(self):
        # Delete the output nodes before next iteration of incremental generation.
        for o in self.outputs:
            o.value_.uses_ = []
        self.outputs = []

class IncreGraphGenerator(GraphGenerator):
    def __init__(self, ops: Iterable[Op], rng: Generator):
        super().__init__(ops, rng)

    def generate(self, status:IncreGenStatus):
        
        if status.need_init:
            # Generate initial input
            init_in = self._gen_input()
            status.inputs.append(init_in)
            status.value_lu.add(init_in.value_)
            status.need_init = False

        if len(status.oprs) >= max_opr_num:
            return None
        
        status.clean_output()

        # Incrementally construct computation graph, with only one op added 
        while True:
            # Choose a value
            value = self._sample_value(list(status.value_lu.values), {})

            # Choose an operator whose first input matches this value
            op = self._sample_op(value)

            # Generate operation vertex
            opr = self._gen_opr(op, value, status.value_lu, status.inputs)
            if opr is None:
                continue
            else:
                status.oprs.append(opr)
                status.last_opr = opr
                break

        # for opr in status.oprs:
        #     print(opr.op_, end=', ')
        # print()
        # for input in status.inputs:
        #     print(input, end=', ')
        # print()

        # Create final graph        
        status.outputs = [Output(v) for v in status.value_lu.values if len(v.uses_) == 0]
        graph = Graph(status.inputs, status.outputs, status.oprs)

        return graph