import os
import json
from typing import List, Dict, Any
from numpy.random import Generator
from tvm import relay
from GenCoG_cl.gencog.graph import Graph, print_relay, GraphMod
from GenCoG_cl.gencog.graph.relay import build_graph
from ..sugar import gen_tensor_value_dict
from ..util import viz2file
from .rule import Rule

class Pattern:
    def __init__(self, gmod: GraphMod, passes:List[str], 
                 codePath:str, idx:int, rule:Rule = None) -> None:
        self.gmod_ = gmod
        self.path_ = codePath
        self.pass_ = passes
        self.idx_ = idx
        self.rule_ = rule

class PatternCorpus:
    def __init__(self, path:str) -> None:
        self.path_ = path
        self.pdir_ = os.path.join(self.path_, '_patterns')      # path to store patterns
        self.patterns_:Dict[int, Pattern] = {}

        if not os.path.exists(self.pdir_):
            os.makedirs(self.pdir_)

        # Init the Corpus from saved files
        else:
            for dn in os.listdir(self.pdir_):
                pattern_dir = os.path.join(self.pdir_, dn)

                # load the code
                codePath = os.path.join(pattern_dir, 'code.txt')
                with open(codePath, 'r') as f:
                    mod = relay.parse(f.read())
                
                gmod = build_graph(mod)

                # load names of triggered passes
                with open(os.path.join(pattern_dir, 'pass.txt'), 'r') as f:
                    passes = [p.strip() for p in f.readlines()]
                
                self.patterns_[int(dn)] = Pattern(gmod, passes, codePath, int(dn))

                # load attributes
                rulePath = os.path.join(pattern_dir, 'rule.json')
                if os.path.exists(rulePath):
                    self.patterns_[int(dn)].rule_ = Rule.load(rulePath)
    
    def register(self, gmod: GraphMod, passes: List[str]):
        # store graph to self.pdir_
        pattern_idx = len(self.patterns_)
        pattern_dir = os.path.join(self.pdir_, str(pattern_idx))
        os.mkdir(pattern_dir)
        code = print_relay(gmod)
        codePath = os.path.join(pattern_dir, 'code.txt')
        with open(codePath, 'w') as f:
            f.write(code)

        # store names of triggered passes
        with open(os.path.join(pattern_dir, 'pass.txt'), 'w') as f:
            f.writelines([pn+'\n' for pn in passes])
        
        # visualize graph
        viz2file(codePath)

        # reload the graph and register the pattern to self.patterns_
        # reloading aims to protect the original graph
        with open(codePath, 'r') as f:
            mod = relay.parse(f.read())
        reloaded_graph = build_graph(mod)
        self.patterns_[pattern_idx] = Pattern(reloaded_graph, passes.copy(), codePath, pattern_idx)

        # store the graph file to the pass folder
        # this step only aims for better observation and debugging
        for pn in passes:
            pass_dir = os.path.join(self.path_, pn)
            if not os.path.exists(pass_dir):
                os.mkdir(pass_dir)
            new_codePath = os.path.join(pass_dir, f'pattern_{pattern_idx}.txt')
            with open(new_codePath, 'w') as f:
                f.write(code)
            
        return self.patterns_[pattern_idx]
                
    def __getitem__(self, p_idx:int):
        return self.patterns_[p_idx]
    
    @property
    def indices(self):
        return sorted(list(self.patterns_.keys()))
    
    @property
    def size(self):
        return len(self.patterns_)