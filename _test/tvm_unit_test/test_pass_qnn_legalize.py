# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Test legalize pass"""
import numpy as np
import tvm
from tvm import te

from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay import transform, analysis
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.testing import run_infer_type


def alpha_equal(x, y):
    """
    Wrapper around alpha equality which ensures that
    the hash function respects equality.
    """
    x = x["main"]
    y = y["main"]
    return tvm.ir.structural_equal(x, y) and tvm.ir.structural_hash(x) == tvm.ir.structural_hash(y)


def run_opt_pass(expr, passes):
    passes = passes if isinstance(passes, list) else [passes]
    mod = tvm.IRModule.from_expr(expr)
    seq = tvm.transform.Sequential(passes)
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    entry = mod["main"]
    return entry if isinstance(expr, relay.Function) else entry.body


def test_qnn_legalize():
    """Test directly replacing an operator with a new one"""

    def before():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")
        y = relay.qnn.op.requantize(
            x,
            input_scale=relay.const(1, "float32"),
            input_zero_point=relay.const(0, "int32"),
            output_scale=relay.const(1, "float32"),
            output_zero_point=relay.const(0, "int32"),
            out_dtype="int8",
        )
        y = relay.Function([x], y)
        return y

    def legalize_qnn_requantize(attrs, inputs, types):
        data = inputs[0]
        data = relay.add(relay.const(0, "int8"), data)
        y = relay.qnn.op.requantize(
            data,
            input_scale=relay.const(1, "float32"),
            input_zero_point=relay.const(0, "int32"),
            output_scale=relay.const(1, "float32"),
            output_zero_point=relay.const(0, "int32"),
            out_dtype="int8",
        )
        return y

    def expected():
        x = relay.var("x", shape=(1, 64, 56, 56), dtype="int8")
        y = relay.add(relay.const(0, "int8"), x)
        z = relay.qnn.op.requantize(
            y,
            input_scale=relay.const(1, "float32"),
            input_zero_point=relay.const(0, "int32"),
            output_scale=relay.const(1, "float32"),
            output_zero_point=relay.const(0, "int32"),
            out_dtype="int8",
        )
        z = relay.Function([x], z)
        return z

    a = before()

    with TempOpAttr("qnn.requantize", "FTVMQnnLegalize", legalize_qnn_requantize):

        # Check that Relay Legalize does not change the graph.
        a = run_opt_pass(a, relay.transform.Legalize())
        b = run_opt_pass(before(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)

        # Check that QNN Legalize modifies the graph.
        a = run_opt_pass(a, relay.qnn.transform.Legalize())
        b = run_opt_pass(expected(), transform.InferType())
        assert tvm.ir.structural_equal(a, b), "Actual = \n" + str(a)


def test_qnn_legalize_qnn_conv2d():
    def _get_mod(data_dtype, kernel_dtype):
        data_shape = (1, 64, 256, 256)
        kernel_shape = (128, 64, 3, 3)
        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        kernel = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)
        func = relay.qnn.op.conv2d(
            data,
            kernel,
            input_zero_point=relay.const(1, "int32"),
            kernel_zero_point=relay.const(1, "int32"),
            input_scale=relay.const(1.0, "float32"),
            kernel_scale=relay.const(1.0, "float32"),
            kernel_size=(3, 3),
            channels=kernel_shape[0],
            strides=(1, 1),
            dilation=(1, 1),
            out_dtype="int32",
            data_layout="NCHW",
            kernel_layout="OIHW",
        )

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)
        return mod

    # Check uint8 x uint8 and int8 x int8 transformation
    for dtype in ("uint8", "int8"):
        mod = _get_mod(dtype, dtype)

        #############################################################
        # Check transformations for platforms with fast Int8 support.
        #############################################################
        # Check that Intel AVX512 (with or w/o VNNI) gets picked up.
        for target in ["llvm -mcpu=skylake-avx512", "llvm -mcpu=cascadelake"]:
            with tvm.target.Target(target):
                mod = relay.transform.InferType()(mod)
                legalized_mod = relay.qnn.transform.Legalize()(mod)
                assert "cast" in legalized_mod.astext() and "qnn.conv2d" in legalized_mod.astext()

        # Since same dtype, there should not be any transformation
        with tvm.target.Target(
            "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        ):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert tvm.ir.structural_equal(mod, legalized_mod)

        ################################################################
        # Check transformations for platforms without fast Int8 support.
        ################################################################
        # Older Intel versions.
        with tvm.target.Target("llvm"):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

        # Older ARM vesions.
        with tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Check uint8 x int8 transformation
    mod = _get_mod("uint8", "int8")
    #############################################################
    # Check transformations for platforms with fast Int8 support.
    #############################################################
    # Check no transformation for Intel AVX512.
    with tvm.target.Target("llvm -mcpu=skylake-avx512"):
        mod = relay.transform.InferType()(mod)
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert tvm.ir.structural_equal(mod, legalized_mod)

    # ARM - so check that transformation has happened.
    with tvm.target.Target(
        "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
    ):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn.conv2d" in legalized_mod.astext()

    ################################################################
    # Check transformations for platforms without fast Int8 support.
    ################################################################
    # Older Intel versions.
    with tvm.target.Target("llvm"):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Older ARM vesions.
    with tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    ###########################################
    # Check transformations for CUDA platforms.
    ###########################################
    with tvm.target.Target("cuda"):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn" in legalized_mod.astext()


def test_qnn_legalize_qnn_dense():
    def _get_mod(data_dtype, kernel_dtype):
        data_shape = (10, 3)
        kernel_shape = (20, 3)
        data = relay.var("data", shape=data_shape, dtype=data_dtype)
        kernel = relay.var("kernel", shape=kernel_shape, dtype=kernel_dtype)
        func = relay.qnn.op.dense(
            data,
            kernel,
            input_zero_point=relay.const(1, "int32"),
            kernel_zero_point=relay.const(1, "int32"),
            input_scale=relay.const(1, "float32"),
            kernel_scale=relay.const(1, "float32"),
            units=kernel_shape[0],
            out_dtype="int32",
        )

        mod = relay.Function(relay.analysis.free_vars(func), func)
        mod = tvm.IRModule.from_expr(mod)
        return mod

    # Check uint8 x uint8 and int8 x int8 transformation
    for dtype in ("uint8", "int8"):
        mod = _get_mod(dtype, dtype)

        #############################################################
        # Check transformations for platforms with fast Int8 support.
        #############################################################
        # Check that Intel AVX512 (with or w/o VNNI) gets picked up.
        for target in ["llvm -mcpu=skylake-avx512", "llvm -mcpu=cascadelake"]:
            with tvm.target.Target(target):
                mod = relay.transform.InferType()(mod)
                legalized_mod = relay.qnn.transform.Legalize()(mod)
                assert "cast" in legalized_mod.astext() and "qnn.dense" in legalized_mod.astext()

        # Since same dtype, there should not be any transformation
        with tvm.target.Target(
            "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
        ):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert tvm.ir.structural_equal(mod, legalized_mod)

        ################################################################
        # Check transformations for platforms without fast Int8 support.
        ################################################################
        # Older Intel versions.
        with tvm.target.Target("llvm"):
            print(mod)
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            print(legalized_mod)
            assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

        # Older ARM vesions.
        with tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"):
            legalized_mod = relay.qnn.transform.Legalize()(mod)
            assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Check uint8 x int8 transformation
    mod = _get_mod("uint8", "int8")
    #############################################################
    # Check transformations for platforms with fast Int8 support.
    #############################################################
    # Check no transformation for Intel AVX512.
    with tvm.target.Target("llvm -mcpu=skylake-avx512"):
        mod = relay.transform.InferType()(mod)
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert tvm.ir.structural_equal(mod, legalized_mod)

    # ARM - so check that transformation has happened.
    with tvm.target.Target(
        "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mattr=+v8.2a,+dotprod"
    ):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn.dense" in legalized_mod.astext()

    ################################################################
    # Check transformations for platforms without fast Int8 support.
    ################################################################
    # Older Intel versions.
    with tvm.target.Target("llvm"):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    # Older ARM vesions.
    with tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu"):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn" not in legalized_mod.astext()

    ###########################################
    # Check transformations for CUDA platforms.
    ###########################################
    with tvm.target.Target("cuda"):
        legalized_mod = relay.qnn.transform.Legalize()(mod)
        assert "cast" in legalized_mod.astext() and "qnn" in legalized_mod.astext()


def test_qnn_legalize_qnn_conv2d_non_scalar_qnn_params():
    """
    Test QNN legalization for qnn.conv2d op for Hexagon target when kernel zero point and kernel
    scale are vectors of scalars.
    """
    data_shape = (1, 29, 16, 16)
    weights_shape = (60, 29, 3, 3)
    O, I = weights_shape[0], weights_shape[1]
    data = relay.var("data", shape=data_shape, dtype="uint8")
    weights = relay.var("weight", shape=weights_shape, dtype="int8")
    data_zp = relay.const(2)
    data_scale = relay.const(0.15)

    def before():
        op = relay.qnn.op.conv2d(
            data,
            weights,
            input_zero_point=data_zp,
            kernel_zero_point=relay.const([1] * O),
            input_scale=data_scale,
            kernel_scale=relay.const([0.17] * O),
            padding=[0, 0, 0, 0],
            channels=O,
            kernel_size=[3, 3],
        )
        return op

    def expected():
        in_diff = 3
        out_diff = 4
        op0 = relay.nn.pad(weights, pad_width=[[0, 0], [0, in_diff], [0, 0], [0, 0]])
        op1 = relay.nn.pad(data, pad_width=[[0, 0], [0, in_diff], [0, 0], [0, 0]])
        op2 = relay.nn.pad(op0, pad_width=[[0, out_diff], [0, 0], [0, 0], [0, 0]])
        op3 = relay.qnn.op.conv2d(
            op1,
            op2,
            input_zero_point=data_zp,
            kernel_zero_point=relay.const([1] * O + [0] * out_diff),
            input_scale=data_scale,
            kernel_scale=relay.const([0.17] * O + [1.0] * out_diff),
            padding=[0, 0, 0, 0],
            channels=(O + out_diff),
            kernel_size=[3, 3],
        )
        op4 = relay.strided_slice(op3, begin=[0, 0, 0, 0], end=[1, 60, 14, 14], strides=[1])
        return op4

    target = tvm.target.hexagon("v68")
    with tvm.target.Target(target):
        a = run_opt_pass(before(), relay.qnn.transform.Legalize())
        b = run_infer_type(expected())
        tvm.ir.assert_structural_equal(a, b)


def test_qnn_legalize_qnn_dense_non_scalar_qnn_params():
    """
    Test QNN legalization for qnn.dense op for Hexagon target when kernel zero point and kernel
    scale are vectors of scalars.
    """
    data_shape = (4, 16)
    weights_shape = (58, 16)
    N = weights_shape[0]
    data = relay.var("data", shape=data_shape, dtype="uint8")
    weights = relay.var("weight", shape=weights_shape, dtype="int8")
    data_zp = relay.const(2)
    data_scale = relay.const(0.15)

    def before():
        wzp = relay.const([1] * N)
        wscale = relay.const([0.17] * N)
        op = relay.qnn.op.dense(data, weights, data_zp, wzp, data_scale, wscale, units=N)
        return op

    def expected():
        diff = 6
        wzp = relay.const([1] * N + [0] * diff)
        wscale = relay.const([0.17] * N + [1.0] * diff)
        op0 = relay.nn.pad(weights, pad_width=[[0, diff], [0, 0]])
        op1 = relay.qnn.op.dense(data, op0, data_zp, wzp, data_scale, wscale, units=(N + diff))
        op2 = relay.strided_slice(op1, begin=[0, 0], end=[data_shape[0], N], strides=[1], axes=None)
        return op2

    target = tvm.target.hexagon("v68")
    with tvm.target.Target(target):
        a = run_opt_pass(before(), relay.qnn.transform.Legalize())
        b = run_infer_type(expected())
        tvm.ir.assert_structural_equal(a, b)


if __name__ == "__main__":
    test_qnn_legalize()
    test_qnn_legalize_qnn_conv2d()
    test_qnn_legalize_qnn_dense()
    test_qnn_legalize_qnn_conv2d_non_scalar_qnn_params()
    test_qnn_legalize_qnn_dense_non_scalar_qnn_params()
