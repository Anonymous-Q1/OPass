import os

tvm_root = '/home/nie/tvm-gcov/tvm/'
gcno_save_dir = '/home/nie/tvm-gcov/tvm-gcno/'

if not os.path.exists(gcno_save_dir):
    os.makedirs(gcno_save_dir)

# Move gcno files to gcno_save_dir for protection
search_list = [tvm_root]
while search_list:
    current_dir = search_list.pop(0)
    for name in os.listdir(current_dir):
        path = os.path.join(current_dir, name)
        if os.path.isdir(path):
            if not os.path.exists(gcno_save_dir + path[len(tvm_root):]):
                os.mkdir(gcno_save_dir + path[len(tvm_root):])
            search_list.append(path)

        if path.endswith('.gcno') and not os.path.exists(gcno_save_dir + path[len(tvm_root):]):
            os.system(f'mv {path} {gcno_save_dir + path[len(tvm_root):]}')

# Copy useful gcno files to the original directory
protect_files = [
    'build/CMakeFiles/tvm_objs.dir/src/relay/parser/parser.cc.gcno', 
    'build/CMakeFiles/tvm_objs.dir/src/relay/collage/collage_partitioner.cc.gcno', 
    'build/CMakeFiles/tvm_objs.dir/src/relay/backend/vm/lambda_lift.cc.gcno', 
    'build/CMakeFiles/tvm_objs.dir/src/relay/backend/vm/removed_unused_funcs.cc.gcno',
]
protect_dirs = [
    'build/CMakeFiles/tvm_objs.dir/src/relay/transforms'
]
for file in protect_files:
    os.system(f'cp {os.path.join(gcno_save_dir, file)} {os.path.join(tvm_root, file)}')
for dir in protect_dirs:
    dir_path = os.path.join(gcno_save_dir, dir)
    des_path = os.path.join(tvm_root, dir)
    for file in os.listdir(dir_path):
        if file.endswith('.gcno'):
            os.system(f'cp {os.path.join(dir_path, file)} {os.path.join(des_path, file)}')
