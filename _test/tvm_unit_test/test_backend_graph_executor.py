import numpy as np
import pytest
from unittest.mock import patch

import tvm
import json
from tvm import relay
from tvm.contrib import graph_executor
from tvm.relay.op import add
import tvm.testing
from tvm.relay.testing import mlp
from tvm import rpc
from tvm.contrib import utils

def test_plan_memory():
    # it is sufficient to cycle through two memories.

    x = relay.var("x", shape=(10,))
    y = relay.var("x", shape=(1,))
    y2 = relay.exp(y)
    # y3 = relay.zeros((1,), dtype='float32')
    z = relay.add(x, y2)
    # z = relay.add(z, y3)
    z = relay.exp(z)
    # z = relay.exp(z)
    # z = relay.exp(z)
    # z = relay.exp(z)
    # z = relay.exp(z)
    z2 = relay.add(z, x)
    # z = relay.Tuple([z, z2])
    # z = relay.concatenate(z, -1)
    # z3 = relay.multiply(z, x)
    # z = relay.add(z2, z3)
    # z2 = relay.add(z, x)
    # z3 = relay.multiply(y, y)
    # z = relay.add(z2, z3)
    # z = relay.reshape(z, (1, -1))
    func = relay.Function([x, y], z)
    mod = tvm.IRModule.from_expr(func)
    mod = relay.transform.InferType()(mod)
    # mod = relay.transform.FuseOps(0)(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)
    memory_plan = relay.backend._backend.GraphPlanMemory(func)
    storage_ids = set()
    device_types = set()
    storage_sizes = {}


    for k, v in memory_plan.expr_to_storage_info.items():
        print('---k---')
        print(k)
        print(isinstance(k, relay.Call))
        print('---v---')
        for x in v.storage_ids:
            print(x)
            print(v.storage_sizes)
        for x in v.storage_ids:
            storage_ids.add(x)
            storage_sizes[x] = v.storage_sizes
        for x in v.device_types:
            device_types.add(x)
    print(storage_sizes, device_types)

    # Current rule requires vars have unique storage id
    # because we don't do inplace, we will need another
    # two alternating temporary space.
    # assert len(storage_ids) == 4, f"found storage_ids: {storage_ids}"
    # assert len(device_types) == 1
    # assert len(storage_sizes) == 4

    # # Check the specific size of each sid
    # assert (
    #     storage_sizes[0][0] == 40
    #     and storage_sizes[1][0] == 4
    #     and storage_sizes[2][0] == 4
    #     and storage_sizes[3][0] == 40
    # )

test_plan_memory()

x = relay.var("x", shape=(10,))
y = relay.var("x", shape=(1,))
y2 = relay.exp(y)
z = relay.add(x, y2)
z = relay.exp(z)
z2 = relay.add(z, x)
z = (z, z2)
z = relay.concatenate(z, -1)
func = relay.Function([x, y], z)
mod = tvm.IRModule.from_expr(func)

def cal_tvm_mem(mod):
    mod = relay.transform.InferType()(mod)
    func = mod["main"]
    mod = relay.transform.InferType()(mod)
    memory_plan = relay.backend._backend.GraphPlanMemory(func)

    storage_ids = set()
    device_types = set()
    storage_sizes = {}

    for k, v in memory_plan.expr_to_storage_info.items():
        for x in v.storage_ids:
            storage_ids.add(x)
            sizes = max(v.storage_sizes)
            if sizes > storage_sizes.get(x, 0):
                storage_sizes[x] = sizes
        for x in v.device_types:
            device_types.add(x)
    
    assert len(device_types) == 1

    totol_sizes = 0
    for k, v in storage_sizes.items():
        totol_sizes += v
    
    return totol_sizes

# print(cal_tvm_mem(mod))