import os
import numpy as np
import pathlib
import json
from PIL import Image
import tarfile
from numpy.random import Generator, PCG64
from typing import List

import tvm
from tvm import relay
from tvm.relay import parser
from tvm.relay.backend import Executor, Runtime
from tvm.contrib.download import download_testdata
from tvm.micro import export_model_library_format
from tvm.relay.op.contrib import cmsisnn
from tvm.micro.testing.utils import create_header_file

def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)

def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}

codePath = '/home/nie/RelayOpt/eval/transformer/transformer.txt'
with open(codePath, 'r') as f:
    code = f.read()
relay_mod = parser.parse(code)

# Generate input parameters
rng = Generator(PCG64(seed=42))
main_fn = relay_mod['main']
inputs = gen_tensor_value_dict(main_fn.params[0:1], rng)
params = gen_tensor_value_dict(main_fn.params[1:], rng)

# We can use TVM native schedules or rely on the CMSIS-NN kernels using TVM Bring-Your-Own-Code (BYOC) capability.
USE_CMSIS_NN = False

# USMP (Unified Static Memory Planning) performs memory planning of all tensors holistically to achieve best memory utilization
DISABLE_USMP = False

# Use the C runtime (crt)
RUNTIME = Runtime("crt")

# We define the target by passing the board name to `tvm.target.target.micro`.
# If your board is not included in the supported models, you can define the target such as:
# TARGET = tvm.target.Target("c -keys=arm_cpu,cpu -mcpu=cortex-m4")
# TARGET = tvm.target.target.micro("stm32l4r5zi")
TARGET = tvm.target.Target("c -keys=cpu -model=host")

# Use the AOT executor rather than graph or vm executors. Use unpacked API and C calling style.
EXECUTOR = tvm.relay.backend.Executor(
    # "aot", {"unpacked-api": True, "interface-api": "c", "workspace-byte-alignment": 8}
    'graph'
)

# Now, we set the compilation configurations and compile the model for the target:
config = {"tir.disable_vectorize": True}
if USE_CMSIS_NN:
    config["relay.ext.cmsisnn.options"] = {"mcpu": TARGET.mcpu}
if DISABLE_USMP:
    config["tir.usmp.enable"] = False

with tvm.transform.PassContext(opt_level=3, config=config):
    if USE_CMSIS_NN:
        # When we are using CMSIS-NN, TVM searches for patterns in the
        # relay graph that it can offload to the CMSIS-NN kernels.
        relay_mod = cmsisnn.partition_for_cmsisnn(relay_mod, params, mcpu=TARGET.mcpu)
    lowered = tvm.relay.build(
        relay_mod, target=TARGET, params=params, runtime=RUNTIME, executor=EXECUTOR
    )
parameter_size = len(tvm.runtime.save_param_dict(lowered.get_params()))
print(f"Model parameter size: {parameter_size}")

# We need to pick a directory where our file will be saved.
# If running on Google Colab, we'll save everything in ``/root/tutorial`` (aka ``~/tutorial``)
# but you'll probably want to store it elsewhere if running locally.

BUILD_DIR = pathlib.Path("/home/nie/RelayOpt/eval/transformer/simu")

BUILD_DIR.mkdir(exist_ok=True)

# Now, we export the model into a tar file:
TAR_PATH = pathlib.Path(BUILD_DIR) / "model.tar"
export_model_library_format(lowered, TAR_PATH)