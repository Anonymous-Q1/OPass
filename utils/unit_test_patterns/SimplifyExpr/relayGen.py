from math import sqrt
import pytest
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass, run_infer_type

import numpy as np

x = relay.var("x", shape=(1, 3, 100, 100), dtype="float32")

def before():
        return relay.add(x, x)



f = before()
       
        

mod = tvm.IRModule.from_expr(f)
            

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
