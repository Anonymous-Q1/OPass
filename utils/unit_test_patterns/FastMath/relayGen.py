import tvm
from tvm.ir import IRModule
from tvm import relay
from tvm.relay.transform import FastMath

x1 = relay.var("x1", shape=(1, 16, 16, 16), dtype="float32")
y1 = relay.exp(x1)
func1 = relay.Function([x1], y1)
mod1 = tvm.IRModule.from_expr(func1)

x2 = relay.var("x2", shape=(1, 16, 16, 16), dtype="float32")
y2 = relay.tanh(x2)
func2 = relay.Function([x2], y2)
mod2 = tvm.IRModule.from_expr(func2)

x3 = relay.var("x3", shape=(1, 16, 16, 16), dtype="float32")
y3 = relay.erf(x3)
func3 = relay.Function([x3], y3)
mod3 = tvm.IRModule.from_expr(func3)

x4 = relay.var("x4", shape=(1, 16), dtype="float32")
y4 = relay.nn.softmax(x4)
func4 = relay.Function([x4], y4)
mod4 = tvm.IRModule.from_expr(func4)

y5 = relay.tanh(relay.exp(x1)) 
func5 = relay.Function([x1],y5)
mod5 = tvm.IRModule.from_expr(func5)

with open('./code.txt', 'w') as f:
    f.write(mod5.astext())
