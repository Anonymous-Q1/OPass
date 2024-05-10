import pytest
import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform
from tvm.relay.prelude import Prelude

def before():
    mod = tvm.IRModule({})
    fn1 = relay.Function([], relay.const(1))
    fn2 = relay.Function([], relay.const(2))
    g1 = relay.GlobalVar("g1")
    g2 = relay.GlobalVar("g2")
    mod[g1] = fn1
    mod[g2] = fn2
    p = relay.var("p", "bool")
    mod["main"] = relay.Function([p], relay.Call(relay.If(p, g1, g2), []))
    return mod

mod = before()


with open('./code.txt', 'w') as f:
    f.write(mod.astext())
