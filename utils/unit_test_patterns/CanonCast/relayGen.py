import tvm
import tvm.testing
from tvm import relay
from tvm.relay import transform

shape = (1,16,7,7)

data = relay.var("data", shape=shape, dtype="int8")
conv_weight = relay.var("weight")
bias1 = relay.var("bias1", shape=(16, 1, 1), dtype="int32")
bias2 = relay.var("bias2", shape=(16, 1, 1), dtype="int32")

x = relay.nn.conv2d(
            data, conv_weight, channels=16, kernel_size=(3, 3), padding=(1, 1), out_dtype="int8"
)
x1 = relay.cast(x, dtype="int32")
y1 = relay.add(x1, bias1)
y2 = relay.add(x1, bias2)
y = relay.add(y1, y2)
func = relay.Function([data, conv_weight, bias1, bias2], y)
mod = tvm.IRModule.from_expr(func)

with open('./code.txt', 'w') as f:
    f.write(mod.astext())
