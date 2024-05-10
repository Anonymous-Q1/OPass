import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay import transform

def before():
    mod = tvm.IRModule()

    x = relay.var("x", shape=(2,))
    i = relay.var("i", shape=(), dtype="int32")
    s = relay.var("s", shape=(2,))
    cond = i < relay.const(10, dtype="int32")

    loop = relay.var("while_loop")
    sb = relay.scope_builder.ScopeBuilder()
    with sb.if_scope(cond):
        ii = i + relay.const(1, dtype="int32")
        ss = s + x
        sb.ret(loop(ii, ss))
    with sb.else_scope():
        sb.ret(s)
    func = relay.Function([i, s], sb.get())

    ret = relay.Let(
        loop, func, loop(relay.const(0, dtype="int32"), relay.zeros(shape=(2,), dtype="float32"))
    )
    mod["main"] = relay.Function([x], ret)
    return mod

mod = before()

     




with open('./code.txt', 'w') as f:
    f.write(mod.astext())
