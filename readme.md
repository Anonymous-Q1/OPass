# Guidelines
## Environments
```
tvm 0.13.dev0
networkx 3.1
numpy 1.24.1
pygraphviz
```
## Orchestrating the passes for a single computation grpah
```
python3 run_transfer_graph.py -p=/path/to/relay/ir.txt
```
## Orchestrating the passes for a set of computation grpahs
```
python3 eval_transfer_graph.py -d=/path/to/directory/of/relay/ir
```
Here the structure of directory is
```
- directory
    - 1
        - code.txt
    - 2
        - code.txt
    ...
```
To reproduce the reaults of RQ1, following
```
mkdir out
cp -rf ReBench ./out/
python3 eval_transfer_graph.py -d=./out/ReBench
```
To optimize lower bound of memory footprints instead of upper bound, replace the `profiler` in `run_transfer_graph.py` from `simu_mem_from_relay` to `serenity_mem_from_relay`.

# Code Structure
```
- OPass
    - _test: the unit test code from TVM's source code.
    - Autotuning: the main implementation to orchestrating passes.
    - GenCoG: the library for generating ReBench.
    - GenCoG_cl: the modified library for analyzing the computation graph when orchestrating passes.
    - ReBench: the benchmark suites.
    - utils: some data for generating benchmarks.
    - eval
        - pattern_table: the patterns in Table III.
        - transformer & resnet: the real-world benchmarks.
    - z_eval_pattern_table: codes for analyzing patterns in eval/pattern_table.
    - run_combgen.py & run_incregen.py: scripts for generating benchmarks.
    - run_transfer_graph.py & eval_transfer_graph.py: scripts for orchestrating passes.
```