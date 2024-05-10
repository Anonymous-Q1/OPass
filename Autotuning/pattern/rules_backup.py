Pattern_Rules = {
'corpus/_patterns/0': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []}
        ],
        'opr': [],
    }
},

'corpus/_patterns/1': {
    "reshape": {
        "input": [
            {"is_anchor": True, "rank_range": [4, 4], "rel_anchor": []}
        ],
        "opr": []
    }
},

'corpus/_patterns/2': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/3': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/4': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/5': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/6': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]'
            ]}
        ],
        'opr': [],
    }
},

'corpus/_patterns/7': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any" if len(x) == 2 else x[1]',
                'lambda x:"any" if len(x) == 3 else x[2]',
                'lambda x:"any" if len(x) == 4 else x[3]',
                'lambda x:"any" if len(x) == 5 else x[4]'
            ]},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any" if len(x) == 2 else x[1]',
                'lambda x:"any" if len(x) == 3 else x[2]',
                'lambda x:"any" if len(x) == 4 else x[3]',
                'lambda x:"any" if len(x) == 5 else x[4]'
            ]},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any" if len(x) == 2 else x[1]',
                'lambda x:"any" if len(x) == 3 else x[2]',
                'lambda x:"any" if len(x) == 4 else x[3]',
                'lambda x:"any" if len(x) == 5 else x[4]'
            ]},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any" if len(x) == 2 else x[1]',
                'lambda x:"any" if len(x) == 3 else x[2]',
                'lambda x:"any" if len(x) == 4 else x[3]',
                'lambda x:"any" if len(x) == 5 else x[4]'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/8': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x: x[0] == 1',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
        ],
        'opr': [
            {'op': 'reshape', 'pos': '000', 'attrs':{
                'newshape': [
                    'lambda x:x[0]',
                    'lambda x:x[2]',
                    'lambda x:x[1]',
                ]
            }}, 
            {'op': 'broadcast_to', 'pos': '100', 'attrs':{
                'shape': [
                    'lambda x:x[1]',
                    'lambda x:x[1]',
                    'lambda x:x[1]',
                ]
            }}
        ],
    }
},

'corpus/_patterns/9': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/10': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:1',
                'lambda x:1',
            ]},
        ],
        'opr': [],
    }
},   # TODO: change 'lambda x:1' to x[2] == x[3]. Method: nest another lambda

'corpus/_patterns/11': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'reshape', 'pos': '000', 'attrs':{
                'newshape': [
                    'lambda x:x[0]',
                    'lambda x:x[1]',
                    'lambda x:-1',
                ]
            }}, 
        ],
    }
},

'corpus/_patterns/12': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 4], 'rel_anchor': [
                'lambda x:x[0]*x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/13': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [5, 5], 'rel_anchor': [
                'lambda x:x[-1] == 4'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/14': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[1]*3',
                'lambda x:x[1]'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/15': {
    "reshape": {
        "input": [
            {"is_anchor": True, "rank_range": [4, 4], "rel_anchor": []},
            {"is_anchor": False, "rank_range": [0, 0], "rel_anchor": []},
            {"is_anchor": False, "rank_range": [0, 0], "rel_anchor": []},
        ],
        "opr": []
    }
},

'corpus/_patterns/16': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:x[-1] == 1'
            ]},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:"any" if len(x) == 1 else x[0]',
                'lambda x:"any" if len(x) == 2 else x[1]',
                'lambda x:"any" if len(x) == 3 else x[2]',
                'lambda x:"any" if len(x) == 4 else x[3]',
                'lambda x:"any" if len(x) == 5 else x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/17': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/18': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 4], 'rel_anchor': [
                'lambda x:x[0]*x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/19': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x: 1 in x'
            ]},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x:[idx for idx, elem in enumerate(x) if elem == 1][0]',
                'lambda x:[idx for idx, elem in enumerate(x) if elem == 1][1]',
                'lambda x:[idx for idx, elem in enumerate(x) if elem == 1][2]',
                'lambda x:[idx for idx, elem in enumerate(x) if elem == 1][3]',
                'lambda x:[idx for idx, elem in enumerate(x) if elem == 1][4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/20': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/21': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/22': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 0], 'rel_anchor': []},
        ],
        'opr': [],
    }
},  # TODO: may support more rank.

'corpus/_patterns/23': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 4], 'rel_anchor': [
                'lambda x:x[0]*x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/24': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]*3'
            ]},
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/25': {
  "reshape": {
    "input": [
      {"is_anchor": True, "rank_range": [5, 5], "rel_anchor": []},
      {"is_anchor": False, "rank_range": [0, 0], "rel_anchor": []},
      {"is_anchor": False, "rank_range": [0, 0], "rel_anchor": []},
      {"is_anchor": False, "rank_range": [0, 0], "rel_anchor": []},
    ],
    "opr": []
  }
},

'corpus/_patterns/26': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:"any" if x[0] % 5 < 4 else None',
                'lambda x:"any" if x[0] % 5 < 3 else None',
                'lambda x:"any" if x[0] % 5 < 2 else None',
                'lambda x:"any" if x[0] % 5 < 1 else None',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/27': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/28': None,

'corpus/_patterns/29': None,

'corpus/_patterns/30': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:len([elem for elem in x if elem > 1]) > 0'
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:"any"'
            ]},
        ],
        'opr': [
            {'op': 'strided_slice', 'pos': '000', 'attrs':{
                'begin': [
                    'lambda x:0 if len([elem for elem in x if elem > 1]) >= 1 else None',
                    'lambda x:0 if len([elem for elem in x if elem > 1]) >= 2 else None',
                    'lambda x:0 if len([elem for elem in x if elem > 1]) >= 3 else None',
                    'lambda x:0 if len([elem for elem in x if elem > 1]) >= 4 else None',
                    'lambda x:0 if len([elem for elem in x if elem > 1]) >= 5 else None',
                ],
                'end': [
                    'lambda x:[elem for elem in x if elem > 1][0] if len([elem for elem in x if elem > 1]) >= 1 else None',
                    'lambda x:[elem for elem in x if elem > 1][1] if len([elem for elem in x if elem > 1]) >= 2 else None',
                    'lambda x:[elem for elem in x if elem > 1][2] if len([elem for elem in x if elem > 1]) >= 3 else None',
                    'lambda x:[elem for elem in x if elem > 1][3] if len([elem for elem in x if elem > 1]) >= 4 else None',
                    'lambda x:[elem for elem in x if elem > 1][4] if len([elem for elem in x if elem > 1]) >= 5 else None',
                ],
                'strides': [
                    'lambda x:1 if len([elem for elem in x if elem > 1]) >= 1 else None',
                    'lambda x:1 if len([elem for elem in x if elem > 1]) >= 2 else None',
                    'lambda x:1 if len([elem for elem in x if elem > 1]) >= 3 else None',
                    'lambda x:1 if len([elem for elem in x if elem > 1]) >= 4 else None',
                    'lambda x:1 if len([elem for elem in x if elem > 1]) >= 5 else None',
                ],
            }}, 
        ],
    }
},

'corpus/_patterns/31': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},],
        'opr': [],
    }
},

'corpus/_patterns/32': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},],
        'opr': [],
    }
},

'corpus/_patterns/33': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x:"any" if len(x) >= 1 else None',
                'lambda x:"any" if len(x) >= 2 else None',
                'lambda x:"any" if len(x) >= 3 else None',
                'lambda x:"any" if len(x) >= 4 else None',
                'lambda x:"any" if len(x) >= 5 else None',
            ]},
        ],
        'opr': [],
    }
},  # TODO: in1 can have different rank with in0

'corpus/_patterns/34': None,

'corpus/_patterns/35': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/36': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x: 1 if len(x) >= 1 else None',
                'lambda x: x[1]',
                'lambda x: x[2]',
                'lambda x: x[3]',
                'lambda x: x[4]',
            ]},
        ],
        'opr': [],
    }
},  # TODO: rel_anchor[0] can be 1 or x[0].

'corpus/_patterns/37': None,

'corpus/_patterns/38': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:64',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
        ],
        'opr': [],
    }
},  # TODO: can be more general.

'corpus/_patterns/39': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:64',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
        ],
        'opr': [],
    }
},  # TODO: can be more general.

'corpus/_patterns/40': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/41': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:64',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
        ],
        'opr': [],
    }
},  # TODO: can be more general.

'corpus/_patterns/42': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
                'lambda x:"any"',
                'lambda x:x[4]',
            ]},
            {'is_anchor': False, 'rank_range': [4, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
                'lambda x:"any"',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/43': None,

'corpus/_patterns/44': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:x[0] % 3 == 0'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/45': None,

'corpus/_patterns/46': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:512',
                'lambda x:x[1]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
        ],
        'opr': [],
    }
},  # TODO: can be more general.

'corpus/_patterns/47': None,

'corpus/_patterns/48': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 5], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/49': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/50': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:1',
                'lambda x:1',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:1',
                'lambda x:1',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:1',
                'lambda x:1',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/51': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:1',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:1',
            ]},
        ],
        'opr': [],
    }
},      # TODO: Check what pass it belong to.

'corpus/_patterns/52': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]%2 == 0'
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:int(x[1]/2)',
                'lambda x:x[1]',
                'lambda x:1',
                'lambda x:1',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/53': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/54': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},      # TODO: Check what pass it belong to.

'corpus/_patterns/55': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[0] + 2',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0] + 2',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:1',
            ]},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:2*(x[0]+2)',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:2*(x[0]+2)',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:1',
            ]},
        ],
        'opr': [
            {'op': 'reshape', 'pos': '00000', 'attrs':{
                'newshape': [
                    'lambda x:1',
                    'lambda x:1',
                    'lambda x:x[0]*(x[0]+2)',
                ]
            }}, 
            {'op': 'reshape', 'pos': '40000', 'attrs':{
                'newshape': [
                    'lambda x:1',
                    'lambda x:1',
                    'lambda x:2*x[0]*(x[0]+2)',
                ]
            }}, 
        ],
    }
},

'corpus/_patterns/56': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/57': {
    "reshape": {
        "input": [
            {"is_anchor": True, "rank_range": [0, 5], "rel_anchor": []}
        ],
        "opr": []
    }
},

'corpus/_patterns/58': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/59': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:1',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:1',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/60': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[0]*2',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/61': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/62': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]',
            ]},
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
            ]},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/63': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]',
            ]},
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
            ]},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/64': None,    # TODO: Global Function.

'corpus/_patterns/65': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/66': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/67': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/68': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/69': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]'
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]'
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]'
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/70': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/71': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/72': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[2]'
            ]},
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/73': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/74': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/75': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/76': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[2]'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/77': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/78': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [5, 5], 'rel_anchor': [
                'lambda x:x[4] == 4',
                'lambda x:x[1] % 2 == 0'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/79': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/80': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/81': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/82': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/83': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/84': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/85': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x: "any" if len(x) >= 1 else None',
                'lambda x: "any" if len(x) >= 2 else None',
                'lambda x: "any" if len(x) >= 3 else None',
                'lambda x: "any" if len(x) >= 4 else None',
                'lambda x: "any" if len(x) >= 5 else None',
            ]},
        ],
        'opr': [],
    }
},  # TODO: can be more general.

'corpus/_patterns/86': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/87': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[-1] % 4 == 0'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/88': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/89': None,    # TODO: support this.

'corpus/_patterns/90': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/91': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/92': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/93': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/94': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:1',
                'lambda x:1',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '000', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }}, 
            {'op': 'nn.conv2d', 'pos': '00010', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }}, 
            {'op': 'nn.conv2d', 'pos': '00011', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }}, 
        ],
    }
},

'corpus/_patterns/95': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/96': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/97': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/98': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[-1] == 1'
            ]},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:1',
                'lambda x:"any"',
            ]},
        ],
        'opr': [],
    }
},    # TODO: LazyGradientInit

'corpus/_patterns/99': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},    # TODO: LazyGradientInit

'corpus/_patterns/100': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/101': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},    # TODO: LazyGradientInit

'corpus/_patterns/102': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},    # TODO: LazyGradientInit

'corpus/_patterns/103': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},    # TODO: LazyGradientInit

'corpus/_patterns/104': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},    # TODO: LazyGradientInit

'corpus/_patterns/105': None,   # TODO: tuple input/

'corpus/_patterns/106': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/107': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/108': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 5], 'rel_anchor': [
                'lambda x:len([d for d in x if d != 1]) != 0'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/109': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[2] % 2 == 0',
                'lambda x:x[3] % 2 == 0',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/110': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/111': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[-1] % 4 == 0'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/112': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1] == 16',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]*2',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]*2',
                'lambda x:3',
                'lambda x:3',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/113': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]*2',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]*2',
                'lambda x:3',
                'lambda x:3',
            ]},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '00', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }},
            {'op': 'nn.conv2d', 'pos': '01', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }},
            {'op': 'nn.conv2d', 'pos': '000000', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }},
            {'op': 'nn.conv2d', 'pos': '000001', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ]
            }},
        ],
    }
},

'corpus/_patterns/114': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/115': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},   # TODO: FlattenAtrousConv

'corpus/_patterns/116': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},   # TODO: FlattenAtrousConv

'corpus/_patterns/117': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/118': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/119': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/120': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/121': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/122': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/123': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/124': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/125': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/126': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/127': None,   # TODO: FoldExplicitPadding

'corpus/_patterns/128': None,   # TODO: FoldExplicitPadding

'corpus/_patterns/129': None,   # TODO: FoldExplicitPadding

'corpus/_patterns/130': None,   # TODO: FoldExplicitPadding

'corpus/_patterns/131': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:x[1]',
                'lambda x:x[1]',
                'lambda x:3',
                'lambda x:3',
            ]},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '00', 'attrs':{
                'channels': [
                    'lambda x:x[1]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/132': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/133': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:"any"',
            ]},
        ],
        'opr': [],
    }
},  # TODO: Can be more general.

'corpus/_patterns/134': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/135': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 2], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [2, 2], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/136': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/137': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/138': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 5], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:x[1]',
                'lambda x:x[2]',
                'lambda x:x[3]',
                'lambda x:x[4]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/139': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/140': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:"any"',
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
            ]},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '0000', 'attrs':{
                'kernel_size': [
                    'lambda x:x[2]',
                    'lambda x:x[3]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/141': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:"any"',
                'lambda x:"any"',
                'lambda x:x[1]',
            ]},
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [1, 1], 'rel_anchor': [
                'lambda x:x[0]',
            ]},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '0000', 'attrs':{
                'kernel_size': [
                    'lambda x:x[2]',
                    'lambda x:x[3]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/142': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},   

'corpus/_patterns/143': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'squeeze', 'pos': '000', 'attrs':{
                'axis': [
                    'lambda x:[idx for idx, d in enumerate(x) if d == 1][0]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/144': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/145': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/146': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/147': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/148': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'strided_slice', 'pos': '000', 'attrs':{
                'begin': [
                    'lambda x:x[0] - x[0]',
                    'lambda x:x[1] - x[1]',
                    'lambda x:x[2] - x[2]',
                    'lambda x:x[3] - x[3]',
                    'lambda x:x[4] - x[4]',
                ],
                'end': [
                    'lambda x:x[0]/2 if x[0]%2==0 else (x[0]+1)/2',
                    'lambda x:x[1]/2 if x[1]%2==0 else (x[1]+1)/2',
                    'lambda x:x[2]/2 if x[2]%2==0 else (x[2]+1)/2',
                    'lambda x:x[3]/2 if x[3]%2==0 else (x[3]+1)/2',
                    'lambda x:x[4]/2 if x[4]%2==0 else (x[4]+1)/2',
                ]
            }},
        ],
    }
},

'corpus/_patterns/149': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/150': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/151': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '000', 'attrs':{
                'kernel_size': [
                    'lambda x:x[2]',
                    'lambda x:x[3]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/152': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/153': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [2, 5], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'reshape', 'pos': '000', 'attrs':{
                'newshape': [
                    'lambda x:x[1]',
                    'lambda x:x[0]',
                    'lambda x:x[2]',
                    'lambda x:x[3]',
                    'lambda x:x[4]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/154': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/155': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [1, 5], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [0, 0], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/156': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/157': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 5], 'rel_anchor': [
                'lambda x:x[3]%2==0'
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/158': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/159': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'nn.conv2d_transpose', 'pos': '000', 'attrs':{
                'kernel_size': [
                    'lambda x:x[2]',
                    'lambda x:x[3]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/160': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [0, 5], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/161': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
            {'is_anchor': False, 'rank_range': [3, 3], 'rel_anchor': [
                'lambda x:x[0]',
                'lambda x:"any"',
                'lambda x:x[2]',
            ]},
        ],
        'opr': [],
    }
},

'corpus/_patterns/162': {
    'reshape':{
        'input': [
            {'is_anchor': False, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:"any"',
                'lambda x:x[1]',
                'lambda x:"any"',
                'lambda x:"any"',
            ]},
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [
            {'op': 'nn.conv2d', 'pos': '000', 'attrs':{
                'kernel_size': [
                    'lambda x:x[2]',
                    'lambda x:x[3]',
                ],
            }},
        ],
    }
},

'corpus/_patterns/163': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [3, 3], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/164': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': []},
        ],
        'opr': [],
    }
},

'corpus/_patterns/165': {
    'reshape':{
        'input': [
            {'is_anchor': True, 'rank_range': [4, 4], 'rel_anchor': [
                'lambda x:len([d for d in x if d%3==0]) != 0'
            ]},
        ],
        'opr': [],
    }
},

}