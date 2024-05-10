from .relay_pass import RelayPass

from typing import List
from tvm.ir.transform import Sequential

class RelaySeq(object):
    def __init__(self) -> None:
        self.passes:List[RelayPass] = []
        self.funcs = []
    
    def append(self, p:RelayPass):
        self.passes.append(p)
        self.funcs.append(p.pass_func)

    def insert(self, index:int, p:RelayPass):
        self.passes.insert(index, p)
        self.funcs.insert(index, p.pass_func)

    @property
    def seq(self):
        return Sequential(self.funcs)
    
    @property
    def info(self):
        return [(p.name, p.params) for p in self.passes]
    
    @property
    def len(self):
        assert len(self.passes) == len(self.funcs)
        return len(self.passes)
    
    def save(self, filePath:str):
        with open(filePath, 'w') as f:
            f.write(f'{str(self.info)}\n')

    def contained(self, request:RelayPass, param_compared = True):
        if param_compared:
            for p in self.passes:
                if p == request:
                    return True
        else:
            for p in self.passes:
                if p.name == request.name:
                    return True
        return False

    def copy(self):
        new_seq = RelaySeq()
        new_seq.passes = self.passes[:]
        new_seq.funcs = self.funcs[:]
        return new_seq

    def slice(self, left:int, right:int):
        self.passes = self.passes[left:right]
        self.funcs = self.funcs[left:right]
    
    def from_info(self, info):
        for name, params in info:
            self.append(RelayPass(name, **params))