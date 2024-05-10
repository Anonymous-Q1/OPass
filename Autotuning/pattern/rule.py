from typing import List, Optional, Dict, Any, cast
import json

'''
Note:
    1. The children of every vertex in Graph must less than or equal to 10, because of the definition of 'pos' in oprR.
'''

class Rule:
    def __init__(self, reshape_rule: Optional['ReshapeRule'] = None) -> None:
        self.reshapeR_ = reshape_rule

    def to_dict(self):
        return {'reshape': self.reshapeR_.to_dict()}

    def dump(self, filePath: str):
        with open(filePath, 'w') as f:
            json.dump(self.to_dict(), f, indent='  ')

    @classmethod
    def from_dict(self, d):
        return Rule(ReshapeRule.from_dict(d['reshape']))

    @classmethod
    def load(cls, filePath: str):
        with open(filePath, 'r') as f:
            rule_dic = json.load(f)
        
        return Rule(ReshapeRule.from_dict(rule_dic['reshape']))
        
class ReshapeRule:
    def __init__(self, input_rule: List['RSInputRule'], opr_rule: List['RSOprRule']) -> None:
        self.inRs_ = input_rule
        self.oprRs_ = opr_rule
    
    def to_dict(self):
        return {'input': [inR.to_dict() for inR in self.inRs_], 'opr': [oprR.to_dict() for oprR in self.oprRs_]}
    
    @classmethod
    def from_dict(cls, d):
        inRs = [RSInputRule.from_dict(i) for i in d['input']]
        oprRs = [RSOprRule.from_dict(o) for o in d['opr']]
        return ReshapeRule(inRs, oprRs)
    
    @property
    def anchor_idx(self):
        for i, inR in enumerate(self.inRs_):
            if inR.is_anchor_:
                return i
        raise Exception(f'Cannot find the anchor in {self.to_dict()}.')

class RSInputRule:
    def __init__(self, is_anchor: bool, rank_range: List[int] = [], rel_anchor: List[str] = []) -> None:
        self.is_anchor_ = is_anchor
        self.rank_range_ = rank_range
        self.rel_anchor_ = rel_anchor
    
    def to_dict(self):
        return {'is_anchor': self.is_anchor_, 'rank_range': self.rank_range_, 'rel_anchor': self.rel_anchor_}
    
    @classmethod
    def from_dict(cls, d):
        return RSInputRule(d['is_anchor'], d['rank_range'], d['rel_anchor'])

class RSOprRule:
    def __init__(self, op: str, pos: str, attrs: Dict[str, List[str]]) -> None:
        '''
        ### Params ###
        op: opr's name
        pos: opr's posistion
        attrs: key is attr's name. value is attr rules related to anchor shape, similar to rel_anchor.
        '''
        self.op_ = op
        self.pos_ = pos
        self.attrs_ = attrs

    def __getitem__(self, attr:str):
        return self.attrs_[attr]

    def to_dict(self):
        return {'op': self.op_, 'pos': self.pos_, 'attrs': self.attrs_ }
    
    @classmethod
    def from_dict(cls, d):
        return RSOprRule(d['op'], d['pos'], d['attrs'])
