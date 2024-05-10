
from numpy.random import Generator
from typing import List

from .relay_pass import RelayPass, RelayPassTable

class RelayPassSelector(object):
    def __init__(self, rng:Generator) -> None:
        self._rng = rng
    
    def generate_params_for_pass(self, pass_name:str, style:str = 'default', **kwargs):
        '''
        Generate parameters to construct the pass function.

        Parameters
        ----------
        pass_name: str
            The name of the pass.
        '''
        param_list: List[RelayPassTable.PassParam]= RelayPassTable.ParamTable[pass_name]
        constructed_params = {}
        for param in param_list:
            # if the parameter is given.
            if param.name in kwargs.keys():
                constructed_params[param.name] = kwargs[param.name]

            # if the parameter is not given.
            else:
                if style == 'default':
                    if param.necessary == False:
                        constructed_params[param.name] = param.default_value
                    else:
                        constructed_params[param.name] = self.random_generate_param(param, pass_name, **kwargs)

                elif style == 'random':
                    try:
                        constructed_params[param.name] = self.random_generate_param(param, pass_name, **kwargs)
                    except Exception as err:
                        if param.necessary == False:
                            constructed_params[param.name] = param.default_value
                        else:
                            raise err

                else:
                    raise Exception(f'Unvalid style {style} for pass params generation.')
                
        # Parameters refinements
        if pass_name == 'EtaExpand':
            if constructed_params['expand_constructor'] == False and constructed_params['expand_global_var'] == False:
                constructed_params[str(self._rng.choice(['expand_constructor', 'expand_global_var']))] = True

        return constructed_params
                
    def random_generate_param(self, param:RelayPassTable.PassParam, pass_name:str, **kwargs):
        if param.name in RelayPassTable.BoolParamTable:
            return bool(self._rng.choice([True, False]))
        
        elif param.name in RelayPassTable.IntParamTable:
            return int(self._rng.integers(low=RelayPassTable.IntParamTable[param.name][0], high=RelayPassTable.IntParamTable[param.name][1], endpoint=True))
        
        elif param.name in RelayPassTable.FloatParamTable:
            return float(self._rng.random() * (RelayPassTable.FloatParamTable[param.name][1] - RelayPassTable.FloatParamTable[param.name][0]) + RelayPassTable.FloatParamTable[param.name][0])
        
        elif param.name in RelayPassTable.OtherParamTable:
            raise Exception(f'Cannot handle complex Param {param.name} for Pass {pass_name}.')

        else:
            raise Exception(f'Cannot generate Param {param.name} for Pass {pass_name}.')

    def wrap_pass(self, pass_name:str, default = True, **kwargs):
        try:
            if default:
                # Reduce the parameter choices to do simplification.
                params = self.generate_params_for_pass(pass_name, 'default', **kwargs)
                if pass_name == 'FuseOps':
                    params['fuse_opt_level'] = 4
                if pass_name == 'SplitArgs':
                    params['max_function_args'] = 3
            else:
                params = self.generate_params_for_pass(pass_name, 'random', **kwargs)
            p = RelayPass(pass_name, **params)
            return p
        except Exception as e:
            raise e

    def random_choice(self, **kwargs):
        while(True):
            try:
                pn = self._rng.choice(RelayPassTable.NameTable)
                params = self.generate_params_for_pass(pn, 'random', **kwargs)
                p = RelayPass(pn, **params)
                return p
            except:
                continue