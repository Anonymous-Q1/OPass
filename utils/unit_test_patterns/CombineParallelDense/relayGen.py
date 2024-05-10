import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform

x = relay.var("x", shape=(2, 32))
w1 = relay.var("w1", shape=(16, 32))
w2 = relay.var("w2", shape=(8, 32))

args = [x, w1, w2]
y1 = relay.nn.dense(x, w1)
y1 = relay.expand_dims(y1, axis=2)

y2 = relay.nn.dense(x, w2)
y2 = relay.expand_dims(y2, axis=2)

y = relay.Tuple((y1, y2))
mod = tvm.IRModule.from_expr(y)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
