import numpy as np
import tvm
import tvm.testing
from tvm import relay
from tvm.relay.prelude import Prelude
from tvm.relay import op, create_executor, transform
from tvm.relay import Var, TypeVar, TupleGetItem, Let, Function, const, RefRead, RefWrite, RefCreate
from tvm.relay import TensorType, Tuple, If, Clause, PatternConstructor, PatternVar, Match
from tvm.relay import GlobalVar, Call
from tvm.relay.transform import gradient
from tvm.relay.testing import make_nat_expr, run_infer_type

def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def tipe(expr):
    return run_opt_pass(expr, [transform.PartialEvaluate(), transform.InferType()])


def dcpe(expr, mod=None, grad=False, ignore_impurity=False):
    passes = [
        transform.PartialEvaluate(),
        transform.InferType(),
        transform.DeadCodeElimination(inline_once=True, ignore_impurity=ignore_impurity),
        transform.InferType(),
    ]
    if grad:
        expr = gradient(run_infer_type(expr))
    if mod:
        assert isinstance(expr, Function)
        mod["main"] = expr
        seq = tvm.transform.Sequential(passes)
        mod = seq(mod)
        return mod["main"]
    return run_opt_pass(expr, passes)

def before():
    a = relay.Var("a")
    b = relay.Var("b")
    clause = relay.Clause(relay.PatternTuple([relay.PatternVar(a), relay.PatternVar(b)]), a + b)
    x = relay.Match(relay.Tuple([relay.const(1), relay.const(1)]), [clause])
    return x
    
def test_const_inline():
    t = relay.TensorType([], "float32")
    d = relay.Var("d", t)
    r = relay.Var("r", relay.RefType(t))
    x = relay.Var("x")
    body = relay.RefRead(r)
    body = Let(x, RefWrite(r, RefRead(r) * RefRead(r)), body)
    body = Let(r, RefCreate(d), body)
    square = Function([d], body)
    return square
   
f = test_const_inline()
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
