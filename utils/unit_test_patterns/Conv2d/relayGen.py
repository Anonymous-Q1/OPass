import tvm
from tvm import relay
from tvm.relay import transform

x_shape = (1,4,16,16)
repeat = 4

x = relay.var("x", shape=x_shape)
in_c = x_shape[1]
out_c = in_c // 2
w = relay.var("w", shape=(out_c, in_c, 1, 1))


args = [x, w]
y = x
for i in range(repeat):
            y1 = relay.nn.conv2d(y, w)
            y2 = relay.nn.conv2d(y, w)
            y = relay.concatenate((y1, y2), axis=1)
f = relay.Function(args, y)
mod = tvm.IRModule.from_expr(f)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
