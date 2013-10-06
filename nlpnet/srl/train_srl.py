# -*- coding: utf-8 -*-

"""
Auxiliary functions for SRL training.
"""

import cPickle
import re
import numpy as np
from itertools import izip
from collections import defaultdict

import config
import utils
from srl import srl_reader

def load_srl_sentences():
    """
    Loads previously pickled sentences with SRL data.
    """
    with open(config.FILES['srl_sentences'], 'rb') as f:
        sents = cPickle.load(f)
    return sents

def create_conll_gold_file():
    """
    Creates a gold standard file in the CoNLL format.
    :param verbs: list of tuples (position, token)
    """
    r = srl_reader.SRLReader(filename=config.FILES['conll_test'])
    r.convert_tags('iobes')
    lines = []
    
    for sentence, predicates in izip(r.sentences, r.predicates):
        sent_lines = []
        
        # get the verbs with their indices
        actual_sentence = sentence[0]
        verbs = [(pred, actual_sentence[pred].word) for pred in predicates]
        
        # get the proposition tags
        props = sentence[1]
        sent_length = len(props[0])
        
        # defaultdict to know what to print in the verbs column
        verb_dict = defaultdict(lambda: '-', verbs)
        
        for i in range(sent_length):
            verb = verb_dict[i]
            args = [utils.convert_iobes_to_bracket(x[i]) for x in props]
            sent_lines.append('\t'.join([verb] + args))
        
        lines.append('%s\n' % '\n'.join(sent_lines))
        
    # add a line break at the end
    result = '\n'.join(lines) 
    with open(config.FILES['srl_gold'], 'wb') as f:
        f.write(result.encode('utf-8'))

def create_reader_srl(args):
    """
    Creates and returns a SRLReader object for the SRL task.
    """
    sents = load_srl_sentences()
    return srl_reader.SRLReader(sents, only_boundaries=args.identify, 
                                only_classify=args.classify,
                                only_predicates=args.predicates)

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

