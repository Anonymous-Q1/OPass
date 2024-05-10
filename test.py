import os
import json
from tvm import relay
from numpy.random import Generator, PCG64
from Autotuning.util import simu_mem_footprint, viz2file, exec_and_compare, load_gmod_from_file
from Autotuning.serenity import simu_mem_serenity
from Autotuning.graph import subgraph_match
from Autotuning.pattern import PatternCorpus, Rule
from GenCoG_cl.gencog.graph import print_relay, build_graph
# def load_graph_from_file(filePath:str, rng:Generator):
#     from tvm import relay
#     from GenCoG_cl.gencog.graph.relay import build_graph
#     with open(filePath, 'r') as f:
#         mod = relay.parse(f.read())
#     main_fn = mod['main']
#     params = gen_tensor_value_dict(main_fn.params[1:], rng)
#     graph = build_graph(mod, params)
#     return graph

# def gen_tensor_value(var, rng: Generator):
#     var_ty = var.checked_type
#     return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)


# def gen_tensor_value_dict(params, rng: Generator):
#     return {var.name_hint: gen_tensor_value(var, rng) for var in params}

rng = Generator(PCG64(seed=1))
# filePath = './utils/unit_test_patterns/test7.txt'
# g = load_graph_from_file(filePath, rng)
# patternPath = './out/combine-20230626-232349/14/21/code.txt'
# p = load_graph_from_file(patternPath, rng)
# graphs = load_graph_from_file(filePath)
# viz2file(filePath)
# with open(filePath + '.txt', 'w') as f:
#     f.write(print_relay(graphs))
# viz2file(filePath + '.txt')
# print(exec_and_compare(filePath, filePath + '.txt'))
# print(simu_mem_footprint(g))
# simu_mem_serenity(g)
# res = subgraph_match(p, g)
# for match_res in res:
#     print(match_res)

# viz2file('./out/test/code.txt')
'''
Reshape rule
'''
corpus_path = '/home/nie/RelayOpt/corpus/'
corpus = PatternCorpus(corpus_path)
idx = 165
pattern_dir = os.path.dirname(corpus[idx].path_)
rule_path = os.path.join(pattern_dir, 'rule.json')
rule = corpus[idx].rule_

# with open(os.path.join(pattern_dir, 'code.txt'), 'r') as f:
#     mod = relay.parse(f.read())
# mod = relay.transform.InferType()(mod)
# mod = relay.transform.FakeQuantizationToInteger()(mod)
# print(mod)

# gmod = build_graph(mod)
# print(print_relay(gmod))
# viz2file('out/combine-20230913-014255/3/code.txt')
# print(simu_mem_footprint(build_graph(mod)['main']), 'mb')
# print(simu_mem_serenity(build_graph(mod)['main']), 'mb')

'''Rule writing'''
assert rule is None, rule.to_dict()

rule_dict = {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:len([d for d in x if d%3==0]) != 0'
            ]},
        ],
        'opr': [],
    }
}
rule = Rule.from_dict(rule_dict)
rule.dump(rule_path)