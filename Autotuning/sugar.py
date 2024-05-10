import os
from time import strftime
from tvm import relay
from numpy.random import Generator, PCG64

def debug_input(print_dir = False, print_type = False):
    '''
    A decorator for debugging function.
    Try to execute func and print the input if failed.

    Params:
        print_dir: bool. Print dir(param) if True.
    '''
    def decorator(func):
        def wrapper(*arg, **kvargs):
            try:
                return func(*arg, **kvargs)
            except Exception as e:
                print('###### Start Debugging ######')
                print(f'Run func {func.__name__} failed.')
                print('The parameters:')

                for a in arg:
                    print('>', a)
                    if print_dir:
                        print('dir:', dir(a))
                    if print_type:
                        print('type:', type(a))

                for k in kvargs:
                    print('>', k, kvargs[k])
                    if print_dir:
                        print('dir:', kvargs[k])
                    if print_type:
                        print('type:', type(kvargs[k]))
                        
                print('####### End Debugging #######')
                raise e
        return wrapper
    return decorator

def gen_tensor_value(var_ty, rng: Generator):
    if isinstance(var_ty, relay.TensorType):
        return rng.uniform(size=[int(d) for d in var_ty.shape]).astype(var_ty.dtype)
    elif isinstance(var_ty, relay.TupleType):
        return tuple([gen_tensor_value(f, rng) for f in var_ty.fields])
    elif isinstance(var_ty, relay.TypeVar):
        shape = (2, 2)
        dtype = 'float32'
        return rng.uniform(size=[int(d) for d in shape]).astype(dtype)
    else:
        raise Exception(f'Cannot handle relay type {type(var_ty)}.')

def gen_tensor_value_dict(params, rng: Generator):
    """
    Generate parameter tensors for relay function parameters.

    Params
    ------
        params: relay.Funtion.params
        rng: Generator(PCG64(random_seed))
    
    Usage
    ------
        >>> rng = Generator(PCG64(seed=random_seed))
        >>> with open(filePath, 'r') as f: 
        >>>     mod = parse(f.read()) 
        >>> main_fn = mod['main'] 
        >>> inputs = gen_tensor_value_dict(main_fn.params, rng) 
    """
    return {var.name_hint: gen_tensor_value(var.checked_type, rng) for var in params}

class Temp:
    def __init__(self, tempdir = '/home/nie/RelayOpt/utils/temp/') -> None:
        """
        Temporary file path generator.
        """
        self.tempdir_ = os.path.join(tempdir, strftime('temp-%Y%m%d-%H%M%S'))
        assert not os.path.exists(self.tempdir_)
        os.mkdir(self.tempdir_)

        self._cnt = 0

    def tempPath(self, suffix):
        """
        Generate a temporary file path with given file suffix.
        """
        self._cnt += 1
        return os.path.join(self.tempdir_, f'{self._cnt}.{suffix}')
    
    def clear(self):
        os.system(f'rm -rf {self.tempdir_}')
