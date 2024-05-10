from tvm import relay

from inspect import signature
from typing import Optional, Any

TVM_Relay_Passes = {
    'AlterOpLayout': relay.transform.AlterOpLayout, 
    'AnnotateSpans': relay.transform.AnnotateSpans, 
    'AnnotateTarget': relay.transform.AnnotateTarget,             # targets: Any, include_non_call_ops: bool = True
    'BackwardFoldScaleAxis': relay.transform.BackwardFoldScaleAxis, 
    'BatchingOps': relay.transform.BatchingOps, 
    'CanonicalizeCast': relay.transform.CanonicalizeCast, 
    'CanonicalizeOps': relay.transform.CanonicalizeOps, 
    'CapturePostDfsIndexInSpans': relay.transform.CapturePostDfsIndexInSpans, 
    'CollagePartition': relay.transform.CollagePartition,           # config: Any, cost_estimator: Any
    'CombineParallelBatchMatmul': relay.transform.CombineParallelBatchMatmul, # min_num_branches: int = 3
    'CombineParallelConv2D': relay.transform.CombineParallelConv2D,      # min_num_branches: int = 3
    'CombineParallelDense': relay.transform.CombineParallelDense,       # min_num_branches: int = 3, to_batch: bool = True
    'Conv2dToSparse': relay.transform.Conv2dToSparse,             # weight_name: Any, weight_shape: Any, layout: Any, kernel_size: Any
    'Conv2dToSparse2': relay.transform.Conv2dToSparse2,            # layout: Any, kernel_size: Any, blocksize: Any, sparsity_threshold: Any
    'ConvertLayout': relay.transform.ConvertLayout,              # desired_layouts: Any
    'DeadCodeElimination': relay.transform.DeadCodeElimination, 
    # relay.transform.Defunctionalization, 
    'DefuseOps': relay.transform.DefuseOps, 
    'DenseToSparse': relay.transform.DenseToSparse,              # weight_name: Any, weight_shape: Any
    'DynamicToStatic': relay.transform.DynamicToStatic, 
    'EliminateCommonSubexpr': relay.transform.EliminateCommonSubexpr,     # fskip: Any
    'EtaExpand': relay.transform.EtaExpand,                  # expand_constructor: bool = False, expand_global_var: bool = False
    'FakeQuantizationToInteger': relay.transform.FakeQuantizationToInteger,  # hard_fail: bool = False, use_qat: bool = False
    'FastMath': relay.transform.FastMath, 
    'FirstOrderGradient': relay.transform.FirstOrderGradient, 
    'FlattenAtrousConv': relay.transform.FlattenAtrousConv, 
    'FoldConstant': relay.transform.FoldConstant,               # fold_qnn: bool = False
    # relay.transform.FoldConstantExpr,           # expr: Any, mod: Any, fold_qnn: bool = False
    'FoldExplicitPadding': relay.transform.FoldExplicitPadding, 
    'FoldScaleAxis': relay.transform.FoldScaleAxis, 
    'ForwardFoldScaleAxis': relay.transform.ForwardFoldScaleAxis, 
    'FuseOps': relay.transform.FuseOps,                    # fuse_opt_level: int = -1
    'InferType': relay.transform.InferType, 
    # relay.transform.InferTypeLocal,             # expr: Any
    'Inline': relay.transform.Inline, 
    'InlineCompilerFunctionsBoundTo': relay.transform.InlineCompilerFunctionsBoundTo, # global_vars: Any
    'LambdaLift': relay.transform.LambdaLift, 
    'LazyGradientInit': relay.transform.LazyGradientInit, 
    'Legalize': relay.transform.Legalize,                   # legalize_map_attr_name: str = "FTVMLegalize"
    # relay.transform.ManifestLifetimes, 
    'MarkCompilerFunctionsAsExtern': relay.transform.MarkCompilerFunctionsAsExtern,  # compiler_filter: str = ""
    'MergeCompilerRegions': relay.transform.MergeCompilerRegions, 
    'MergeComposite': relay.transform.MergeComposite,             # pattern_table: Any
    'OutlineCompilerFunctionsWithExistingGlobalSymbols': relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols,  # compiler_filter: str = ""
    'PartialEvaluate': relay.transform.PartialEvaluate, 
    'PartitionGraph': relay.transform.PartitionGraph,             # mod_name: str = "default", bind_constants: bool = True
    'PlanDevices': relay.transform.PlanDevices,                # config: Any
    'RemoveUnusedFunctions': relay.transform.RemoveUnusedFunctions,      # entry_functions: Any | None = None
    'SimplifyExpr': relay.transform.SimplifyExpr, 
    'SimplifyFCTranspose': relay.transform.SimplifyFCTranspose,        # target_weight_name: Any
    'SimplifyInference': relay.transform.SimplifyInference, 
    'SplitArgs': relay.transform.SplitArgs,                  # max_function_args: Any
    'ToANormalForm': relay.transform.ToANormalForm,
    'ToBasicBlockNormalForm': relay.transform.ToBasicBlockNormalForm,
    # relay.transform.ToCPS,
    'ToGraphNormalForm': relay.transform.ToGraphNormalForm,
    'ToMixedPrecision': relay.transform.ToMixedPrecision,
    # relay.transform.DivToMul(),
}


class RelayPass(object):
    def __init__(self, name, **kwargs) -> None:
        self.name = name
        self.pass_ = RelayPassTable.FuncTable[name]
        self.params = kwargs
        self.pass_func = self.pass_(**kwargs)

    def __eq__(self, __o: object) -> bool:
        return (isinstance(__o, RelayPass) and self.name == __o.name and self.params == __o.params)
            
class RelayPassTable:
    NameTable = [
        'AlterOpLayout', 
        'AnnotateSpans', 
        # 'AnnotateTarget', 
        'BackwardFoldScaleAxis', 
        'BatchingOps', 
        'CanonicalizeCast', 
        'CanonicalizeOps', 
        # 'CapturePostDfsIndexInSpans', 
        # 'CollagePartition', 
        'CombineParallelBatchMatmul', 
        'CombineParallelConv2D', 
        'CombineParallelDense', 
        # 'Conv2dToSparse', 
        # 'Conv2dToSparse2', 
        # 'ConvertLayout', 
        'DeadCodeElimination', 
        'DefuseOps', 
        # 'DenseToSparse', 
        'DynamicToStatic', 
        'EliminateCommonSubexpr', 
        'EtaExpand', 
        'FakeQuantizationToInteger', 
        'FastMath', 
        # 'FirstOrderGradient', 
        'FlattenAtrousConv', 
        'FoldConstant', 
        'FoldExplicitPadding', 
        'FoldScaleAxis', 
        'ForwardFoldScaleAxis', 
        'FuseOps', 
        'InferType', 
        'Inline', 
        # 'InlineCompilerFunctionsBoundTo', 
        'LambdaLift', 
        'LazyGradientInit', 
        'Legalize', 
        'MarkCompilerFunctionsAsExtern', 
        'MergeCompilerRegions', 
        # 'MergeComposite', 
        'OutlineCompilerFunctionsWithExistingGlobalSymbols', 
        'PartialEvaluate', 
        'PartitionGraph', 
        # 'PlanDevices', 
        'RemoveUnusedFunctions', 
        'SimplifyExpr', 
        # 'SimplifyFCTranspose', 
        'SimplifyInference', 
        'SplitArgs', 
        'ToANormalForm',
        'ToBasicBlockNormalForm',
        'ToGraphNormalForm',
        'ToMixedPrecision',
    ]

    FuncTable = {
        'AlterOpLayout': relay.transform.AlterOpLayout, 
        'AnnotateSpans': relay.transform.AnnotateSpans, 
        'AnnotateTarget': relay.transform.AnnotateTarget, 
        'BackwardFoldScaleAxis': relay.transform.BackwardFoldScaleAxis, 
        'BatchingOps': relay.transform.BatchingOps, 
        'CanonicalizeCast': relay.transform.CanonicalizeCast, 
        'CanonicalizeOps': relay.transform.CanonicalizeOps, 
        'CapturePostDfsIndexInSpans': relay.transform.CapturePostDfsIndexInSpans, 
        'CollagePartition': relay.transform.CollagePartition, 
        'CombineParallelBatchMatmul': relay.transform.CombineParallelBatchMatmul, 
        'CombineParallelConv2D': relay.transform.CombineParallelConv2D,  
        'CombineParallelDense': relay.transform.CombineParallelDense,  
        'Conv2dToSparse': relay.transform.Conv2dToSparse,          
        'Conv2dToSparse2': relay.transform.Conv2dToSparse2,       
        'ConvertLayout': relay.transform.ConvertLayout,           
        'DeadCodeElimination': relay.transform.DeadCodeElimination, 
        'DefuseOps': relay.transform.DefuseOps, 
        'DenseToSparse': relay.transform.DenseToSparse,            
        'DynamicToStatic': relay.transform.DynamicToStatic, 
        'EliminateCommonSubexpr': relay.transform.EliminateCommonSubexpr,    
        'EtaExpand': relay.transform.EtaExpand,              
        'FakeQuantizationToInteger': relay.transform.FakeQuantizationToInteger, 
        'FastMath': relay.transform.FastMath, 
        'FirstOrderGradient': relay.transform.FirstOrderGradient, 
        'FlattenAtrousConv': relay.transform.FlattenAtrousConv, 
        'FoldConstant': relay.transform.FoldConstant,              
        'FoldExplicitPadding': relay.transform.FoldExplicitPadding, 
        'FoldScaleAxis': relay.transform.FoldScaleAxis, 
        'ForwardFoldScaleAxis': relay.transform.ForwardFoldScaleAxis, 
        'FuseOps': relay.transform.FuseOps,             
        'InferType': relay.transform.InferType, 
        'Inline': relay.transform.Inline, 
        'InlineCompilerFunctionsBoundTo': relay.transform.InlineCompilerFunctionsBoundTo,
        'LambdaLift': relay.transform.LambdaLift, 
        'LazyGradientInit': relay.transform.LazyGradientInit, 
        'Legalize': relay.transform.Legalize,    
        'MarkCompilerFunctionsAsExtern': relay.transform.MarkCompilerFunctionsAsExtern,  
        'MergeCompilerRegions': relay.transform.MergeCompilerRegions, 
        'MergeComposite': relay.transform.MergeComposite,          
        'OutlineCompilerFunctionsWithExistingGlobalSymbols': relay.transform.OutlineCompilerFunctionsWithExistingGlobalSymbols,  
        'PartialEvaluate': relay.transform.PartialEvaluate, 
        'PartitionGraph': relay.transform.PartitionGraph,           
        'PlanDevices': relay.transform.PlanDevices,          
        'RemoveUnusedFunctions': relay.transform.RemoveUnusedFunctions,    
        'SimplifyExpr': relay.transform.SimplifyExpr, 
        'SimplifyFCTranspose': relay.transform.SimplifyFCTranspose,     
        'SimplifyInference': relay.transform.SimplifyInference, 
        'SplitArgs': relay.transform.SplitArgs,                
        'ToANormalForm': relay.transform.ToANormalForm,
        'ToBasicBlockNormalForm': relay.transform.ToBasicBlockNormalForm,
        'ToGraphNormalForm': relay.transform.ToGraphNormalForm,
        'ToMixedPrecision': relay.transform.ToMixedPrecision,
    }

    class PassParam:
        def __init__(self, name:str, necessary:bool, default_value:Optional[Any] = None) -> None:
            '''
            Parameters
            ----------
            name: str
                The name of the parameter.
            necessary: bool
                Whether this parameter is necessary to construct the pass function.
            default_value: any
                The default value of the parameter.
            '''
            self.name = name
            self.necessary = necessary
            self.default_value = default_value

    ParamTable = {
        'AlterOpLayout': [], 
        'AnnotateSpans': [], 
        'AnnotateTarget': [PassParam('targets', True), PassParam('include_non_call_ops', False, True)], 
        'BackwardFoldScaleAxis': [], 
        'BatchingOps': [], 
        'CanonicalizeCast': [], 
        'CanonicalizeOps': [], 
        'CapturePostDfsIndexInSpans': [], 
        'CollagePartition': [PassParam('config', True), PassParam('cost_estimator', False, None)], 
        'CombineParallelBatchMatmul': [PassParam('min_num_branches', False, 3)], 
        'CombineParallelConv2D': [PassParam('min_num_branches', False, 3)], 
        'CombineParallelDense': [PassParam('min_num_branches', False, 3), PassParam('to_batch', False, True)], 
        'Conv2dToSparse': [PassParam('weight_name', True), PassParam('weight_shape', True), PassParam('layout', True), PassParam('kernel_size', True)], 
        'Conv2dToSparse2': [PassParam('layout', True), PassParam('kernel_size', True), PassParam('blocksize', True), PassParam('sparsity_threshold', True)], 
        'ConvertLayout': [PassParam('desired_layouts', True)], 
        'DeadCodeElimination': [PassParam('inline_once', False, False), PassParam('ignore_impurity', False, False)], 
        'DefuseOps': [], 
        'DenseToSparse': [PassParam('weight_name', True), PassParam('weight_shape', True)], 
        'DynamicToStatic': [], 
        'EliminateCommonSubexpr': [PassParam('fskip', False, None)], 
        'EtaExpand': [PassParam('expand_constructor', False, False), PassParam('expand_global_var', False, False)], 
        'FakeQuantizationToInteger': [PassParam('hard_fail', False, False), PassParam('use_qat', False, False)], 
        'FastMath': [], 
        'FirstOrderGradient': [], 
        'FlattenAtrousConv': [], 
        'FoldConstant': [PassParam('fold_qnn', False, False)], 
        'FoldExplicitPadding': [], 
        'FoldScaleAxis': [], 
        'ForwardFoldScaleAxis': [], 
        'FuseOps': [PassParam('fuse_opt_level', False, -1)], 
        'InferType': [], 
        'Inline': [], 
        'InlineCompilerFunctionsBoundTo': [PassParam('global_vars', True)], 
        'LambdaLift': [], 
        'LazyGradientInit': [], 
        'Legalize': [PassParam('legalize_map_attr_name', False, 'FTVMLegalize')], 
        'MarkCompilerFunctionsAsExtern': [PassParam('compiler_filter', False, '')], 
        'MergeCompilerRegions': [], 
        'MergeComposite': [PassParam('pattern_table', True)], 
        'OutlineCompilerFunctionsWithExistingGlobalSymbols': [PassParam('compiler_filter', False, '')], 
        'PartialEvaluate': [], 
        'PartitionGraph': [PassParam('mod_name', False, 'default'), PassParam('bind_constants', False, True)], 
        'PlanDevices': [PassParam('config', True)], 
        'RemoveUnusedFunctions': [PassParam('entry_functions', False, None)], 
        'SimplifyExpr': [], 
        'SimplifyFCTranspose': [PassParam('target_weight_name', True)], 
        'SimplifyInference': [], 
        'SplitArgs': [PassParam('max_function_args', True)], 
        'ToANormalForm': [],
        'ToBasicBlockNormalForm': [],
        'ToGraphNormalForm': [],
        'ToMixedPrecision': [PassParam('mixed_precision_type', False, 'float16'), PassParam('missing_op_mode', False, 1)], # mixed_precision_type="float16", missing_op_mode=1
    }

    BoolParamTable = [
        'include_non_call_ops', 'to_batch', 'inline_once', 'ignore_impurity', 'expand_constructor', 'expand_global_var',
        'hard_fail', 'use_qat', 'fold_qnn', 'bind_constants', 
    ]

    IntParamTable = {
        'min_num_branches': (2, 5), 
        'kernel_size': (1, 10),
        'blocksize': (1, 10),
        'fuse_opt_level': (0, 5),
        'max_function_args': (1, 10),
        'missing_op_mode': (0, 2),
    }

    FloatParamTable = {
        'sparsity_threshold': (0, 1),
    }

    OtherParamTable = [
        'targets', 'config', 'cost_estimator', 'weight_name', 'weight_shape', 'layout', 'desired_layouts', 'fskip',
        'global_vars', 'legalize_map_attr_name', 'compiler_filter', 'pattern_table', 'compiler_filter', 'mod_name',
        'entry_functions', 'target_weight_name', 'mixed_precision_type',
    ]

    SrcTable = {
        # src/relay/transform/
        'alter_op_layout.cc': ['AlterOpLayout'],
        'annotate_target.cc': ['AnnotateTarget'], 
        # 'annotate_texture_storage.cc': ['AnnotateMemoryScope'],
        # 'auto_scheduler_layout_rewrite.cc': ['AutoSchedulerLayoutRewrite'], 
        'canonicalize_cast.cc': ['CanonicalizeCast'], 
        'canonicalize_ops.cc': ['CanonicalizeOps'],
        'capture_postdfsindex_in_spans.cc': ['CapturePostDfsIndexInSpans'],
        'combine_parallel_batch_matmul.cc': ['CombineParallelBatchMatmul'],
        'combine_parallel_conv2d.cc': ['CombineParallelConv2D'], 
        'combine_parallel_dense.cc': ['CombineParallelDense'],
        # 'combine_parallel_op.cc': None,
        # 'combine_parallel_op_batch.cc': ['CombineParallelOpBatch'],
        'compiler_function_utils.cc': ['MarkCompilerFunctionsAsExtern', 'InlineCompilerFunctionsBoundTo', 'OutlineCompilerFunctionsWithExistingGlobalSymbols'],
        'convert_layout.cc': ['ConvertLayout'],
        'convert_sparse_conv2d.cc': ['Conv2dToSparse', 'Conv2dToSparse2'],
        'convert_sparse_dense.cc': ['DenseToSparse'],
        'dead_code.cc': ['DeadCodeElimination'],
        # 'de_duplicate.cc': None, 
        # 'defunctionalization.cc': None, 
        'defuse_ops.cc': ['DefuseOps'], 
        # 'device_aware_visitors.cc': None, 
        # 'device_domains.cc': None, 
        'device_planner.cc': ['PlanDevices'],
        # 'div_to_mul.cc': ['DivToMul'],
        'dynamic_to_static.cc': ['DynamicToStatic'],
        'eliminate_common_subexpr.cc': ['EliminateCommonSubexpr'],
        'eta_expand.cc': ['EtaExpand'],
        # 'expr_subst.cc': None, 
        'fake_quantization_to_integer.cc': ['FakeQuantizationToInteger'],
        'fast_math.cc': ['FastMath'],
        'first_order_gradient.cc': ['FirstOrderGradient'],
        'flatten_atrous_conv.cc': ['FlattenAtrousConv'],
        'fold_constant.cc': ['FoldConstant'], 
        'fold_explicit_padding.cc': ['FoldExplicitPadding'],
        'fold_scale_axis.cc': ['FoldScaleAxis', 'BackwardFoldScaleAxis', 'ForwardFoldScaleAxis'],
        # 'forward_rewrite.cc': None, 
        'fuse_ops.cc': ['FuseOps'], 
        # 'higher_order_gradient.cc': None, 
        # 'infer_layout_utils.cc': None, 
        'inline.cc': ['Inline'], 
        # 'label_ops.cc': ['LabelOps'], 
        'lazy_gradient_init.cc': ['LazyGradientInit'],
        'legalize.cc': ['Legalize'],
        # 'memory_alloc.cc': ['ManifestAlloc'], 
        'merge_compiler_regions.cc': ['MergeCompilerRegions'], 
        'merge_composite.cc': ['MergeComposite'], 
        # 'meta_schedule_layout_rewrite.cc': ['MetaScheduleLayoutRewrite'], 
        'partial_eval.cc': ['PartialEvaluate'], 
        'partition_graph.cc': ['PartitionGraph'], 
        # 'remove_standalone_reshapes.cc': ['RemoveStandaloneReshapes'], 
        'simplify_expr.cc': ['SimplifyExpr'],   # , 'SimplifyExprPostAlterOp'
        'simplify_fc_transpose.cc': ['SimplifyFCTranspose'], 
        'simplify_inference.cc': ['SimplifyInference'], 
        'split_args.cc': ['SplitArgs'], 
        # 'target_hooks.cc': ['RelayToTIRTargetHook'], 
        'to_a_normal_form.cc': ['ToANormalForm'], 
        'to_basic_block_normal_form.cc': ['ToBasicBlockNormalForm'], 
        # 'to_cps.cc': ['ToCPS', 'UnCPS'], 
        'to_graph_normal_form.cc': ['ToGraphNormalForm'], 
        'to_mixed_precision.cc': ['ToMixedPrecision'],
        'type_infer.cc': ['InferType'],

        # src/relay/parser/
        'parser.cc': ['AnnotateSpans'], 
        # src/relay/collage/
        'collage_partitioner.cc': ['CollagePartition'], 
        # src/relay/backend/vm/
        'lambda_lift.cc': ['LambdaLift'], 
        'removed_unused_funcs.cc': ['RemoveUnusedFunctions'],
    }

# print(len(RelayPassTable.SrcTable))

# print(len(RelayPassTable.FuncTable))
# for func in RelayPassTable.FuncTable:
#     params = signature(RelayPassTable.FuncTable[func]).parameters
#     if len(params) != 0:
#         print(func, params)

# Check if the pass name list of RelayPassTable.SrcTable is equal to RelayPassTable.NameTable
# for k in RelayPassTable.SrcTable:
#     for n in RelayPassTable.SrcTable[k]:
#         if n not in RelayPassTable.NameTable:
#             print(n)
# print('----')
# SrcNameTable = [n for k in RelayPassTable.SrcTable for n in RelayPassTable.SrcTable[k]]
# for n in RelayPassTable.NameTable:
#     if n not in SrcNameTable:
#         print(n)