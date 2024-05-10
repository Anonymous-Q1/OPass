import numpy as np
import pytest

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass
import tvm.testing
import tvm.topi.testing

channel_size = 16

x = relay.var("x", shape=(16, channel_size))
softmax = relay.nn.softmax(x)
out = relay.cast(softmax, "float16")
f = relay.Function([x], out)
mod = tvm.IRModule.from_expr(f)


with open('./code.txt', 'w') as f:
    f.write(mod.astext())
