import tvm
from tvm import relay

def get_mod():
        mod = tvm.IRModule({})
        x0 = relay.var("x0", shape=(3, 5))
        y0 = relay.var("y0", shape=(3, 5))
        fn0 = relay.Function([x0, y0], x0 * y0)
        fn0 = fn0.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        fn0 = fn0.with_attr("Compiler", "aa")
        g0 = relay.GlobalVar("g0")
        mod[g0] = fn0

        x1 = relay.var("x1", shape=(3, 5))
        y1 = relay.var("y1", shape=(3, 5))
        fn1 = relay.Function([x1, y1], x1 + g0(x1, y1))
        g1 = relay.GlobalVar("g1")
        mod[g1] = fn1

        x2 = relay.var("x2", shape=(3, 5))
        y2 = relay.var("y2", shape=(3, 5))
        fn2 = relay.Function([x2, y2], x2 - g1(x2, y2))
        fn2 = fn2.with_attr("Inline", tvm.tir.IntImm("int32", 1))
        g2 = relay.GlobalVar("g2")
        mod[g2] = fn2
        return mod

mod = get_mod()
            

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
