import numpy as np
import os
# import psutil
# import gc
# from memory_profiler import profile

# @profile
# def test():
#     a=np.full(shape=(600, 700), fill_value=99.0)
#     return a

# if __name__ == '__main__':

#     a=test()

#     print('A: %.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))
#     del a
#     gc.collect()
#     print('B: %.2f MB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024))

# from guppy import hpy
# hp = hpy()
# x = np.ones([1000,1000])
# hp.setrelheap()
# y = x
# print(hp.heap())

import tvm
from tvm import relay
from tvm.relay import transform
# x = relay.var("x", shape=(12, 57, 56), dtype="float32")
# y = relay.var("y", shape=(128, 56, 56), dtype="float32")
# d = relay.var("d", shape=(3, 3), dtype="float16")
# c = relay.const(0.1, dtype="float32")
# w = relay.var("w", shape=(4, 4, 3), dtype="float32")
# v = relay.var("v", shape=(4, 4, 3), dtype="float32")
# y = relay.transpose(x, axes=[3, 0, 1, 2])
# y = relay.transpose(x, axes=[0, 1, 2, 3])
# x = relay.ones(shape=(2, 2, 1, 3), dtype="float32")
# y = relay.ones(shape=(2, 2, 1, 3), dtype="float32")
# y = relay.ones_like(x)

# z = relay.nn.batch_matmul(x, y)

# an example
x = relay.var("x", shape=[1, 5, 5, 4], dtype="float32")
w = relay.ones([3, 3, 4, 1], dtype='float32')

x1 = relay.transpose(x, axes=[0, 1, 2, 3])
# x1 = relay.zeros(relay.shape_of(x1))
# x1 = relay.zeros_like(x1)
x1 = relay.nn.conv2d(
        x1,
        w,
        padding=[2, 2, 2, 2],
        dilation=[2, 2],
        groups=4,
        channels=4,
        kernel_size=[3, 3],
        data_layout="NHWC",
        kernel_layout="HWOI",
    )
# x1 = relay.nn.relu(x1)
x1 = relay.layout_transform(x1, "NCHW", "NCHW")
# x1 = relay.squeeze(x1)


# y1 = relay.add(x, w)
# y1 = relay.nn.relu(y1)
# y1 = relay.zeros(relay.shape_of(x), dtype='float32')
# y1 = expected_expr = relay.nn.conv2d(
#         x,
#         w,
#         padding=[2, 2, 2, 2],
#         dilation=[2, 2],
#         groups=4,
#         channels=4,
#         kernel_size=[3, 3],
#         data_layout="NHWC",
#         kernel_layout="HWOI",
#     )
y1 = relay.nn.space_to_batch_nd(x, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
y1 = relay.nn.conv2d(
    y1,
    w,
    padding=[0, 0, 0, 0],
    groups=4,
    channels=4,
    kernel_size=[3, 3],
    data_layout="NHWC",
    kernel_layout="HWOI",
)
y1 = relay.nn.batch_to_space_nd(y1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])

z = relay.add(x1, y1)

func = relay.Function([x], z)
before = tvm.IRModule.from_expr(func)
before = transform.InferType()(before)
print(transform.FuseOps(4)(before))

se = transform.SimplifyExpr()(before)
se_ecs = transform.EliminateCommonSubexpr()(se)

ecs = transform.EliminateCommonSubexpr()(before)

fac = transform.FlattenAtrousConv()(before)
fac_ecs = transform.EliminateCommonSubexpr()(fac)

se_fac = transform.FlattenAtrousConv()(se)
se_fac_ecs = transform.EliminateCommonSubexpr()(se_fac)

# d2s = transform.DynamicToStatic()(before)
# d2s_se = transform.SimplifyExpr()(d2s)

# print(before)
# print(se_fac_ecs)
# print(se_ecs)

# x = relay.var("x", shape=[1, 4, 16, 16], dtype="float32")
# w1 = relay.var("w1", shape=[4, 4, 3, 3], dtype="float32")
# w2 = relay.var("w2", shape=[4, 4, 3, 3], dtype="float32")
# x1 = relay.nn.conv2d(x, w1)
# x2 = relay.nn.conv2d(x, w2)
# x = relay.var("x", shape=[1, 5, 5, 4], dtype="float32")
# w1 = relay.var("w1", shape=[3, 3, 4, 1], dtype="float32")
# w2 = relay.var("w2", shape=[3, 3, 4, 1], dtype="float32")
# x1 = relay.nn.conv2d(
#         x,
#         w1,
#         padding=[2, 2, 2, 2],
#         dilation=[2, 2],
#         groups=4,
#         channels=4,
#         kernel_size=[3, 3],
#         data_layout="NHWC",
#         kernel_layout="HWOI",
#     )
# x2 = relay.nn.conv2d(
#         x,
#         w2,
#         padding=[2, 2, 2, 2],
#         dilation=[2, 2],
#         groups=4,
#         channels=4,
#         kernel_size=[3, 3],
#         data_layout="NHWC",
#         kernel_layout="HWOI",
#     )
# z = relay.Tuple((x1, x2))
# func = relay.Function([x, w1, w2], z)
# before = tvm.IRModule.from_expr(func)
# before = transform.InferType()(before)
# cpc = transform.CombineParallelConv2D(min_num_branches=2)(before)
# print(before)
# print(cpc)

# x = relay.var("x", shape=[1, 4, 16, 16], dtype="float32")
# w = relay.var("w", shape=[4, 4, 1, 1], dtype="float32")
# y1 = relay.nn.space_to_batch_nd(x, block_shape=[2, 2], paddings=[[2, 3], [2, 3]])
# y1 = relay.nn.conv2d(
#     y1,
#     w,
#     padding=[0, 0, 0, 0],
#     groups=4,
#     channels=4,
#     kernel_size=[3, 3],
#     data_layout="NHWC",
#     kernel_layout="HWOI",
# )
# y1 = relay.nn.batch_to_space_nd(y1, block_shape=[2, 2], crops=[[0, 1], [0, 1]])


# with open('./out/test/code.txt', 'w') as f:
#     f.write(mod.astext())


# motivating example
# def conv(data):
#     y = relay.nn.conv2d(data, relay.var("w"), kernel_size=(3, 3), padding=(1, 1), channels=16)
#     return relay.nn.relu(data=y)

# def inception_like(data):
#     c0 = conv(data)
#     c1 = conv(data)
#     return relay.concatenate((c0, c1), axis=1)

# def before(dshape):
#     x = relay.var("x", shape=dshape)
#     in1 = inception_like(x)
#     in2 = inception_like(in1)
#     return relay.Function(relay.analysis.free_vars(in2), in2)

# orig = before([1, 3, 64, 64])
# fuse = relay.transform.FuseOps(4)(relay.transform.InferType()(tvm.IRModule.from_expr(orig)))
# fuse_tomix = relay.transform.ToMixedPrecision()(fuse)
# tomix = relay.transform.ToMixedPrecision()(relay.transform.InferType()(tvm.IRModule.from_expr(orig)))
# tomix_fuse = relay.transform.FuseOps(4)(tomix)

# default_seq_4 = tvm.ir.transform.Sequential(
#     [
#         relay.transform.RemoveUnusedFunctions(),
#         relay.transform.ToBasicBlockNormalForm(),
#         relay.transform.Legalize(),
#         relay.transform.SimplifyInference(),
#         relay.transform.EliminateCommonSubexpr(),
#         relay.transform.CombineParallelConv2D(),
#         relay.transform.CombineParallelDense(),
#         relay.transform.CombineParallelBatchMatmul(),

#         relay.transform.FoldConstant(),
#         relay.transform.FoldScaleAxis(),
#         relay.transform.SimplifyExpr(),
#         relay.transform.CanonicalizeCast(),
#         relay.transform.CanonicalizeOps(),
#         relay.transform.FlattenAtrousConv(),
#         relay.transform.AlterOpLayout(),
#         relay.transform.SimplifyExpr(),
        
#         # relay.transform.ToMixedPrecision(),
#         relay.transform.FastMath(),
#         # relay.transform.FoldConstant(),
#         # relay.transform.SplitArgs(10), # 25~50
#         # relay.transform.FuseOps(4),
#     ],
# )
# default = default_seq_4(relay.transform.InferType()(tvm.IRModule.from_expr(orig)))
# print(default)
# with open('./out/test/code.txt', 'w') as f:
#     f.write(tomix_fuse.astext())

x = relay.var("x", shape=[3], dtype="float32")
y = relay.var("y", shape=[1, 5, 5, 4], dtype="float32")
z = relay.expand_dims(y, axis=1, num_newaxis=1)
func = relay.Function([x, y], z)
mod = tvm.IRModule.from_expr(func)
print(transform.InferType()(mod))
relay.take()