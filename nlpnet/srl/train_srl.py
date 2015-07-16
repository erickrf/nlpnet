# -*- coding: utf-8 -*-

"""
Auxiliary functions for SRL training.
"""

import re
import numpy as np


def init_transitions_simplified(tag_dict):
    """
    This function initializes a tag transition table containing only
    the boundaries IOBES.
    """
    tags = sorted(tag_dict, key=tag_dict.get)
    transitions = []
    
    for tag in tags:
        if tag in 'OES':
            trans = lambda x: 0 if x in 'BOS' else -1000
        elif tag in 'IB':
            trans = lambda x: 0 if x in 'IE' else -1000
        else:
            raise ValueError('Unexpected tag: %s' % tag)
        
        transitions.append([trans(next_tag) for next_tag in tags])
    
    # initial transition
    trans = lambda x: 0 if x in 'BOS' else -1000
    transitions.append([trans(next_tag) for next_tag in tags])
    
    return np.array(transitions, np.float)


def init_transitions(tag_dict, scheme):
    """
    This function initializes the tag transition table setting 
    very low values for impossible transitions.
     
    :param tag_dict: The tag dictionary mapping tag names to the
        network output number.
    :param scheme: either iob or iobes.
    """
    scheme = scheme.lower()
    assert scheme in ('iob', 'iobes'), 'Unknown tagging scheme: %s' % scheme
    transitions = []
    
    # since dict's are unordered, let's take the tags in the correct order
    tags = sorted(tag_dict, key=tag_dict.get)
    
    # transitions between tags
    for tag in tags:
        
        if tag == 'O':
            # next tag can be O, V or any B
            trans = lambda x: 0 if re.match('B|S|V', x) \
                                else -1 if x == 'O' else -1000
        
        elif tag[0] in 'IB':
            block = tag[2:]
            if scheme == 'iobes':
                # next tag can be I or E (same block)
                trans = lambda x: 0 if re.match('(I|E)-%s' % block, x) else -1000
            else:
                # next tag can be O, I (same block) or B (new block)
                trans = lambda x: 0 if re.match('I-%s' % block, x) or re.match('B-(?!%s)' % block, x) \
                                    else -1 if x == 'O' else -1000
        
        elif tag[0] in 'ES':
            # next tag can be O, S (new block) or B (new block)
            block = tag[2:]
            trans = lambda x: 0 if re.match('(S|B)-(?!%s)' % block, x) \
                                else -1 if x == 'O' else -1000

        else:
            raise ValueError('Unknown tag: %s' % tag)
        
        transitions.append([trans(next_tag) for next_tag in tags])  
    
    # starting tag
    # it can be O or any B/S
    trans = lambda x: 0 if x[0] in 'OBS' else -1000
    transitions.append([trans(next_tag) for next_tag in tags])
    
    return np.array(transitions, np.float)

