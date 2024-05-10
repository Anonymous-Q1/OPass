import tvm
from tvm import relay
from tvm.relay import transform

b =2
i = 3
j = 5
k =4

x = relay.var("x", shape=(b, i, k))
w1 = relay.var("w1", shape=(b, j, k))
w2 = relay.var("w2", shape=(b, j, k))
w3 = relay.var("w3", shape=(b, j, k))


args = [x, w1, w2, w3]
y1 = relay.nn.batch_matmul(x, w1)
y2 = relay.nn.batch_matmul(x, w2)
y3 = relay.nn.batch_matmul(x, w3)
y = relay.Tuple((y1, y2, y3))
f = relay.Function(args, y)
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
