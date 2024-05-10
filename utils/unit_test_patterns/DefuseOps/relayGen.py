import numpy
import pytest
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.testing import run_opt_pass

conv_layer1_weight = relay.nn.Constant(
        tvm.nd.array(numpy.ndarray(shape=(64, 7, 7, 3), dtype="float32"))
)
conv_layer2_weight = relay.nn.Constant(
        tvm.nd.array(numpy.ndarray(shape=(64, 3, 3, 64), dtype="float32"))
)

def fused_conv2d_batch_norm(w):
        data = relay.var("data", shape=(1, 224, 224, 3))
        bn_gamma0 = relay.var("bn_gamma0", relay.TensorType((64,), "float32"))
        bn_beta0 = relay.var("bn_beta0", relay.TensorType((64,), "float32"))
        bn_mmean0 = relay.var("bn_mean0", relay.TensorType((64,), "float32"))
        bn_mvar0 = relay.var("bn_var0", relay.TensorType((64,), "float32"))
        c0 = relay.nn.conv2d(
            data,
            w,
            strides=(2, 2),
            padding=(3, 3, 3, 3),
            channels=64,
            kernel_size=(7, 7),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
        )
        c1 = relay.nn.batch_norm(c0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)
        c2 = c1[0]
        return relay.Function(relay.analysis.free_vars(c2), c2)

def fused_conv2d_batch_norm_relu(z):
        data2 = relay.var("data2", shape=(1, 56, 56, 64))
        bn_gamma0 = relay.var("bn_gamma0", relay.TensorType((64,), "float32"))
        bn_beta0 = relay.var("bn_beta0", relay.TensorType((64,), "float32"))
        bn_mmean0 = relay.var("bn_mean0", relay.TensorType((64,), "float32"))
        bn_mvar0 = relay.var("bn_var0", relay.TensorType((64,), "float32"))
        c0 = relay.nn.conv2d(
            data2,
            z,
            padding=(1, 1, 1, 1),
            channels=64,
            kernel_size=(3, 3),
            data_layout="NHWC",
            kernel_layout="OHWI",
            out_layout="NHWC",
        )
        c1 = relay.nn.batch_norm(c0, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0, axis=3)
        c2 = c1[0]
        c3 = relay.nn.relu(data=c2)
        return relay.Function(relay.analysis.free_vars(c3), c3)

def fused_max_pool2d():
        data1 = relay.var("data1", shape=(1, 112, 112, 64))
        a1 = relay.nn.max_pool2d(
            data1,
            pool_size=(3, 3),
            strides=(2, 2),
            padding=(1, 1, 1, 1),
            layout="NHWC",
            out_layout="NHWC",
        )
        return relay.Function(relay.analysis.free_vars(a1), a1)

def fused_add_relu():
        data1 = relay.var("data1", shape=(1, 56, 56, 64))
        data2 = relay.var("data2", shape=(1, 56, 56, 64))
        a0 = relay.add(data1, data2)
        a1 = relay.nn.relu(a0)
        return relay.Function(relay.analysis.free_vars(a1), a1)

data = relay.var("data", shape=(1, 3, 224, 224))
data1 = relay.layout_transform(data, src_layout="NCHW", dst_layout="NHWC")
bn_gamma0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
bn_beta0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
bn_mmean0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
bn_mvar0 = relay.const(tvm.nd.array(numpy.ndarray(shape=(64,), dtype="float32")))
a0 = fused_conv2d_batch_norm(conv_layer1_weight)
a1 = fused_max_pool2d()
a2 = fused_conv2d_batch_norm_relu(conv_layer2_weight)
a3 = fused_add_relu()
y0 = relay.Call(a0, [data1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0])
y1 = relay.Call(a1, [y0])
y2 = relay.Call(a2, [y1, bn_gamma0, bn_beta0, bn_mmean0, bn_mvar0])
y3 = relay.Call(a3, [y1, y2])

f = relay.Function(relay.analysis.free_vars(y3), y3)
mod = tvm.IRModule.from_expr(f)


with open('./code.txt', 'w') as f:
    f.write(mod.astext())
