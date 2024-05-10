'''
This is a demo for seperate optimization.
This demo create an illustrating example, which need a 3-round optimization 
-- (SimplifyExpr, FlattenAtrousConv, EliminateCommonSubexpr) to achieving the
minimum memory footprint, while the seperate optimization only needs 2-round.
The seperate optimization simultaneously optimize two sub-graph by SimplifyExpr
and FlattenAtrousConv respectively in the first round. Then it apply ECS in 
the second round.
'''
import os
from typing import cast
import tvm
from tvm import relay
from tvm.relay import transform

from GenCoG_cl.gencog.graph import build_graph, print_relay, Graph, Input, Output, Operation, Value, GraphMod
from Autotuning.util import viz2file

root = '/home/nie/RelayOpt/eval/demo_seperate/'

def origin():
    """This function create the original computation graph to be optimized."""
    x = relay.var("x", shape=[1, 5, 5, 4], dtype="float32")
    w = relay.ones([3, 3, 4, 1], dtype='float32')

    # This sub-graph need SimplifyExpr to simplify.
    x1 = relay.transpose(x, axes=[0, 1, 2, 3])
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
    x1 = relay.layout_transform(x1, "NCHW", "NCHW")

    # This sub-graph need FlattenAtrousConv to simplify.
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
    return before

def save_mod(mod, code_dir):
    if os.path.exists(code_dir):
        os.system(f'rm -rf {code_dir}')
    os.mkdir(code_dir)
    code_path = os.path.join(code_dir, 'code.txt')
    with open(code_path, 'w') as f:
        f.write(mod.astext())
    viz2file(code_path)

def no_seperate_opt(orig):
    opt_root = os.path.join(root, 'no_seperate')
    if os.path.exists(opt_root):
        os.system(f'rm -rf {opt_root}')
    os.mkdir(opt_root)

    # First round - SimplifyExpr
    first = transform.SimplifyExpr()(orig)
    save_mod(first, os.path.join(opt_root, '1st'))

    # Second round - FlattenAtrousConv
    second = transform.FlattenAtrousConv()(first)
    save_mod(second, os.path.join(opt_root, '2nd'))

    # Third round - EliminateCommonSubexpr
    third = transform.EliminateCommonSubexpr()(second)
    save_mod(third, os.path.join(opt_root, '3rd'))

def seperate_opt(orig):
    opt_root = os.path.join(root, 'seperate')
    if os.path.exists(opt_root):
        os.system(f'rm -rf {opt_root}')
    os.mkdir(opt_root)

    # First round - SimplifyExpr & FlattenAtrousConv
    graph_mod = build_graph(orig)   # Parse relay mod to computation graph
    graph = graph_mod['main']
    
    extracted_oprs = []
    '''
    Extract left sub-graph for SimplifyExpr
    '''
    connection_points_1 = {'inputs': [], 'outputs': []}
    inputs = []
    oprs = []
    outputs = []

    # 1th opr - transpose
    for opr in graph.oprs_:
        if opr.op_.name_ == 'transpose':
            first_opr = opr
            break
    input1 = Input(first_opr.inputs_[0].type_, False)   # Create a new input
    inputs.append(input1)
    connection_points_1['inputs'].append(first_opr.inputs_[0])
    oprs.append(first_opr)
    first_opr.inputs_[0].uses_ = []                     # Delete the edge from the big graph to sub graph
    first_opr.inputs_ = [input1.value_]

    # 2nd opr - cov2d
    second_opr = cast(Operation, first_opr.outputs_[0].uses_[0])
    input2 = Input(second_opr.inputs_[1].type_, False)
    inputs.append(input2)
    connection_points_1['inputs'].append(second_opr.inputs_[1])
    oprs.append(second_opr)
    second_opr.inputs_[1].uses_ = []
    second_opr.inputs_ = [second_opr.inputs_[0], input2.value_]

    # 3rd opr - layout_transform
    third_opr = cast(Operation, second_opr.outputs_[0].uses_[0])
    output1 = Output(Value(third_opr.outputs_[0].type_, third_opr))    # Create a new output
    outputs.append(output1)
    connection_points_1['outputs'].append(third_opr.outputs_[0])
    oprs.append(third_opr)
    third_opr.outputs_[0].def_ = None                   # Delete the edge from the sub graph to big graph
    third_opr.outputs_ = [output1.value_]
    
    sub_graph_1 = Graph(inputs, outputs, oprs)
    sub_mod_1 = print_relay(GraphMod({'main': sub_graph_1}))
    sub_mod_1 = relay.parse(sub_mod_1)
    save_mod(sub_mod_1, os.path.join(opt_root, 'extracted_left'))

    extracted_oprs += oprs

    '''
    Extract right sub-graph for FlattenAtrousConv
    '''
    connection_points_2 = {'inputs': [], 'outputs': []}
    inputs = []
    oprs = []
    outputs = []

    # 1th opr - nn.space_to_batch_nd
    for opr in graph.oprs_:
        if opr.op_.name_ == 'nn.space_to_batch_nd':
            first_opr = opr
            break
    input1 = Input(first_opr.inputs_[0].type_, False)   # Create a new input
    inputs.append(input1)
    connection_points_2['inputs'].append(first_opr.inputs_[0])
    oprs.append(first_opr)
    first_opr.inputs_[0].uses_ = []                     # Delete the edge from the big graph to sub graph
    first_opr.inputs_ = [input1.value_]

    # 2nd opr - cov2d
    second_opr = cast(Operation, first_opr.outputs_[0].uses_[0])
    input2 = Input(second_opr.inputs_[1].type_, False)
    inputs.append(input2)
    connection_points_2['inputs'].append(second_opr.inputs_[1])
    oprs.append(second_opr)
    second_opr.inputs_[1].uses_ = []
    second_opr.inputs_ = [second_opr.inputs_[0], input2.value_]

    # 3rd opr - nn.batch_to_space_nd
    third_opr = cast(Operation, second_opr.outputs_[0].uses_[0])
    output1 = Output(Value(third_opr.outputs_[0].type_, third_opr))    # Create a new output
    outputs.append(output1)
    connection_points_2['outputs'].append(third_opr.outputs_[0])
    oprs.append(third_opr)
    third_opr.outputs_[0].def_ = None                   # Delete the edge from the sub graph to big graph
    third_opr.outputs_ = [output1.value_]
    
    sub_graph_2 = Graph(inputs, outputs, oprs)
    sub_mod_2 = print_relay(GraphMod({'main': sub_graph_2}))
    sub_mod_2 = relay.parse(sub_mod_2)
    save_mod(sub_mod_2, os.path.join(opt_root, 'extracted_right'))
        
    extracted_oprs += oprs
    oprs_big_graph = [opr for opr in graph.oprs_ if opr not in extracted_oprs]
    ins_big_graph = graph.inputs_
    outs_big_graph = graph.outputs_

    '''
    Optimize extracted sub-graphs
    '''
    opt_mod_1 = transform.SimplifyExpr()(sub_mod_1)
    opt_mod_2 = transform.FlattenAtrousConv()(sub_mod_2)
    save_mod(opt_mod_1, os.path.join(opt_root, '1st_opted_left'))
    save_mod(opt_mod_2, os.path.join(opt_root, '1st_opted_right'))
    
    '''
    Connect the left optimized sub-graph to original graph
    '''
    opt_graph_1 = build_graph(opt_mod_1)['main']
    # Inputs
    for inp, cv in zip(opt_graph_1.inputs_, connection_points_1['inputs']):
        iv = inp.value_
        cv = cast(Value, cv)
        assert iv.type_.shape_ == cv.type_.shape_
        
        for opr in iv.uses_:
            opr = cast(Operation, opr)
            opr.inputs_ = [v if v != iv else cv for v in opr.inputs_]
            cv.uses_.append(opr)
        iv.uses_ = []
    
    # Outputs
    for outp, cv in zip(opt_graph_1.outputs_, connection_points_1['outputs']):
        ov = outp.value_
        cv = cast(Value, cv)
        assert ov.type_.shape_ == cv.type_.shape_

        opr = cast(Operation, ov.def_)
        opr.outputs_ = [v if v != ov else cv for v in opr.outputs_]
        ov.def_ = None
        cv.def_ = opr
    
    oprs_big_graph += opt_graph_1.oprs_

    '''
    Connect the right optimized sub-graph to original graph
    '''
    opt_graph_2 = build_graph(opt_mod_2)['main']
    # Inputs
    for inp, cv in zip(opt_graph_2.inputs_, connection_points_2['inputs']):
        iv = inp.value_
        cv = cast(Value, cv)
        assert iv.type_.shape_ == cv.type_.shape_
        
        for opr in iv.uses_:
            opr = cast(Operation, opr)
            opr.inputs_ = [v if v != iv else cv for v in opr.inputs_]
            cv.uses_.append(opr)
        iv.uses_ = []
    
    # Outputs
    for outp, cv in zip(opt_graph_2.outputs_, connection_points_2['outputs']):
        ov = outp.value_
        cv = cast(Value, cv)
        assert ov.type_.shape_ == cv.type_.shape_

        opr = cast(Operation, ov.def_)
        opr.outputs_ = [v if v != ov else cv for v in opr.outputs_]
        ov.def_ = None
        cv.def_ = opr

    oprs_big_graph += opt_graph_1.oprs_
    
    big_graph = Graph(ins_big_graph, outs_big_graph, oprs_big_graph)
    first = print_relay(GraphMod({'main': big_graph}))
    first = relay.parse(first)
    save_mod(first, os.path.join(opt_root, '1st'))
    
    # Second round - EliminateCommonSubexpr
    second = transform.EliminateCommonSubexpr()(first)
    save_mod(second, os.path.join(opt_root, '2nd'))


def main():
    if os.path.exists(root):
        os.system(f'rm -rf {root}')
    os.mkdir(root)

    # Create original model
    orig = origin()
    orig_dir = os.path.join(root, 'origin')
    save_mod(orig, orig_dir)
    
    # No-seperate optimization
    no_seperate_opt(orig)

    # Seperate optimization
    seperate_opt(orig)



if __name__ == '__main__':
    main()