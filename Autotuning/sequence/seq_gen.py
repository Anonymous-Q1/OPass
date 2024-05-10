from numpy.random import Generator
from typing import Optional

from .relay_seq import RelaySeq
from .pass_gen import RelayPassSelector

class RandomRelaySeq(object):
    def __init__(self, rng:Generator) -> None:
        self._rng = rng

    def generate(self, max_len:int = 5) -> RelaySeq:
        '''
        Randomly generate a relay pass sequence.

        Parameters
        ----------
        max_len: int
            Max length of the pass sequence.

        Returns
        -------
        ret: tvm.ir.transform.Sequential
            A relay pass sequence.
        '''

        pass_selector = RelayPassSelector(self._rng)
        relay_seq = RelaySeq()
        while relay_seq.len < max_len:
            new_pass = pass_selector.random_choice()

            # Add condition pass before this new pass.
            if new_pass.name in ['FuseOps', 'CanonicalizeCast']:
                precond_pass = pass_selector.wrap_pass('SimplifyInference')
                if not relay_seq.contained(precond_pass, param_compared=False):
                    relay_seq.append(precond_pass)
                
            relay_seq.append(new_pass)

        return relay_seq