# -*- coding: utf-8 -*-

"""
This file contains functions for reading input (training) data.
There are functions for dealing specifically with CoNLL format data 
and others for simpler formats.
"""

import re
from itertools import izip

from attributes import Token

PRE_CONTRACTIONS = ['em', 'a', 'para', 'por', 'de', 'por', 'com', 'lhe']
POS_CONTRACTIONS = ['o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 
                    'me', 'te', 'se', 'nos', 'vos', 
                    'esse', 'essa', 'isso', 'este', 'esta', 'isto',
                    'esses', 'essas', 'estes', 'estas', 
                    'aquele', 'aquela', 'aquilo',
                    'aqueles', 'aquelas']

class ConllPos(object):
    """
    Dummy class for storing the position of each field in a
    CoNLL data file.
    """
    id = 0
    word = 1
    lemma = 2
    pos = 3
    morph = 4
    parse = 7
    pred = 8

chunks = ['NP', 'VP', 'ADVP', 'PP', 'ADJP', 'CL']

def read_plain_srl(filename):
    """
    Reads an SRL file and returns the training data. The file
    should be divided in columns, separated by tabs and/or whitespace.
    First column: tokens
    Second column: - (hyphen) for non-predicates, anything else for predicates.
    Third and next columns: the SRL IOBES tags for each token concerning
    each predicate (3rd column for 1st predicate, 4th for the 2nd, and
    so on).
    
    :returns: a list of tuples in the format (tokens, tags, predicates)
    """
    sentences = []
    
    with open(filename, 'rb') as f:
        token_num = 0
        sentence = []
        tags = []
        predicates = []
        
        for line in f:
            line = unicode(line, 'utf-8').strip()
            
            if line == '':
                # last sentence ended
                sentences.append((sentence, tags, predicates))
                sentence = []
                tags = []
                predicates = []
                token_num = 0
                continue
            
            parts = line.split()
            token = Token(parts[0].strip())
            sentence.append(token)
            
            # check if this is a predicate
            if parts[1].strip() != '-':
                predicates.append(token_num)
            
            # initialize the expected roles
            if tags == []:
                num_preds = len(parts) - 2
                tags = [[] for _ in range(num_preds)]
            
            for i, role in enumerate(parts[2:]):
                # the SRL tags
                tags[i].append(role)
            
            token_num += 1
    
    if sentence != []:
        sentences.append((sentence, tags, predicates))
    
    return sentences
    

def verify_chunk_tag(new_tag, expected_tag):
    """
    Compares the expected chunk tag with the new found one, and 
    returns the correct tag.
    """
    
    # new clause chunks are very rare, we collapse them into a single chunk
    if new_tag in ('FCL', 'ACL', 'ICL'):
        new_tag = 'CL'
    
    if new_tag == expected_tag:
        return expected_tag
    
    # NP's inside a PP don't get an own chunk, nor adjectives or adverbs
    # a coordination may appear inside an NP
    if expected_tag == 'PP' and new_tag in ('NP', 'ADJP', 'ADVP', 'CU'):
        return expected_tag
    
    # adjectives inside NPs don't get an own chunk, nor adverbs that may be inside it
    # a coordination may appear inside an NP
    if expected_tag == 'NP' and new_tag in ('ADJP', 'ADVP', 'CU'):
        return expected_tag
    
    if new_tag not in chunks:
        return 'O'
    
    # same for adverbs inside ADJPs
    if expected_tag == 'ADJP' and new_tag == 'ADVP':
        return expected_tag
    
    return new_tag

def get_chunk_tag(word, parse, expected_block): 
    """
    Examines a given entry in the CoNLL format and returns a tuple
    (chunk_tag, expected_chunk_block). Note that the chunk tag is already 
    encoded in IOB, while the expected block is not.
    """
    if not re.search('[\w%]', word):
        # punctuation
        chunk_tag = 'O'
        expected_block = 'O'
    elif parse[0] == '*':
        # continue inside the block
        chunk_tag = expected_block
        if chunk_tag != 'O':
            chunk_tag = 'I-%s' % chunk_tag
    else:
        # find all tags starting here
        new_levels = re.findall('\w+', parse)
        a_priori_expected_block = expected_block
        
        # I believe it is safer to check every tag transition in the new levels
        for new_tag in new_levels:
            chunk_tag = verify_chunk_tag(new_tag, expected_block)
            expected_block = chunk_tag
        
        if chunk_tag != 'O':
            if chunk_tag == a_priori_expected_block:
                chunk_tag = 'I-%s' % chunk_tag
            else:
                chunk_tag = 'B-%s' % chunk_tag
    
    # reached the end of a constituent
    if ')' in parse:
        expected_block = 'O'
    
    return (chunk_tag, expected_block)


def get_chunks(tree, previous_node='', dominating_node='', new_block=False):
    """
    Traverses a tree extracting chunk information.
    
    :param tree: A syntactic tree.
    :param previous_node: the parent of the subtree passed as 
        first argument.
    :param dominating_node: the label of the chunk dominating
        the subtree so far.
    :param new_block: whether a new block (a new chunk) must 
        start if the new tag is the same as the previous dominating one.
    :returns: a list of (tag, chunk_tag) tuples.
    """
    new_node = tree.node
    node = verify_chunk_tag(new_node, dominating_node)
    
    # a new block starts here, with different tags
    new_block = (new_block and node == dominating_node) or (node != dominating_node)
    
    # or a new block starts with the same tag (except in coordinations)
    new_block = new_block or (new_node == dominating_node and 
                              new_node != previous_node and previous_node != 'CU')
    
    tokens_tags = []
    
    for subtree in tree:
        
        try:
            subtree.node
        
        except AttributeError:
            # this is a leaf
            if not re.search('(?u)\w', subtree) and (node == 'CL' or new_block):
                # punctuation connected with the head or starting a new block
                tokens = [subtree]
                tags = ['O']
                new_block = True
            
            else:
                # split multiwords
                tokens = subtree.split('_')
                if node == 'O':
                    tags = ['O'] * len(tokens) 
                else: 
                    if new_block:
                        tags = ['B-%s' % node] + ['I-%s' % node] * (len(tokens) - 1) 
                        new_block = False
                    else:
                        tags = ['I-%s' % node] * len(tokens)
                
            for token, tag in izip(tokens, tags):
                tokens_tags.append((token, tag))
            
        else:
            subtree_tokens = get_chunks(subtree, new_node, node, new_block)
            
            # if the subtree was a PP, it must be clear that a PP ended.
            # Elsewise, if we are on a child of a PP that 
            # has a PP child, things get messy. 
            for _, tag in subtree_tokens:
                if tag == 'B-PP':
                    new_block = True
                    node = verify_chunk_tag(new_node, 'O')
                    break 
            tokens_tags.extend(subtree_tokens)
    
    return tokens_tags


def read_conll(iterable, read_srl=True):
    """
    Reads a sentence from a sequence of lines in a CoNLL format file.
    
    :returns: if read_srl is True, returns a list of tuples, where each
        one has the format:
        ([token1, token2, ...], [[tag-for-pred1, tag-for-pred1, ...],
                                 [tag-for-pred2, tag-for-pred2, ...]],
         [index-of-pred1, index-of-pred2, ...])
        
        Tags are repeated, NOT in IOBES format.
        If read_srl is False, returns a list of sentences.
    """
    from nltk import Tree
    
    sentences = []
    sentence = []
    instances = None
    num_preds = None
    predicates = []
    token_number = 0
    
    # used to build syntactic trees
    tree_strings = []
    
    for line in iterable:
        uline = unicode(line, 'utf-8')  
        data = uline.split()
        if len(data) <= 1:
            # this is an empty line after a sentence
            
            # build the syntactic tree and attribute each token's chunk
            tree = Tree.parse(' '.join(tree_strings), brackets='[]')
            token_chunks = get_chunks(tree)
            for j, (token, (word, chunk)) in enumerate(izip(sentence, token_chunks)):
                
                assert token.word == word,  \
                "Syntactic and semantic analyses got different words: %s and %s" % (token.word, word)
                
                token.chunk = chunk
                sentence[j] = token
            
            if read_srl:
                sentences.append((sentence, instances, predicates))
                instances = None
                predicates = []
                token_number = 0
            else:
                sentences.append(sentence)
            
            num_preds = None
            tree_strings = []
            sentence = []
            continue
        
        if instances is None and read_srl:
            # initializes each instance as an empty list
            num_preds = len(data) - ConllPos.pred - 1
            instances = [[] for _ in xrange(num_preds)]
            expected_role = ['O'] * num_preds
        
        word = data[ConllPos.word]
        lemma = data[ConllPos.lemma].lower()
        pos = data[ConllPos.pos].lower()
        parse = data[ConllPos.parse]
        is_predicate = data[ConllPos.pred] != '-'
        
        # lemmas for punctuation are listed as -
        if lemma == '-':
            lemma = word
        
        # Syntactic tree
        
        # to avoid messing nltk.Tree string parser, we use [] as tree brackets
        # instead of the default (). This is done because "(" and ")" appear as 
        # separate tokens, while "["and "]" do not.
        tree_string = parse.replace('(', '[').replace(')', ']')
        # treat "broken" constituents like VP- and -VP as normal VPs
        tree_string = tree_string.replace('-', '')
        tree_string = tree_string.replace('*', ' %s ' % word)
        tree_strings.append(tree_string)
        
        # if it's a predicate, add to the list of predicates
        # we must check it before appending the tokens
        # because multiword tokens may mess up the count
        if read_srl and is_predicate:
            predicates.append(token_number)
        
        # split multiwords
        splitted = zip(word.split('_'), lemma.split('_'))
        num_parts = len(splitted)
        for word_part, lemma_part in splitted:
            token = Token(word_part, pos=pos, lemma=lemma_part)
            sentence.append(token)
            token_number += 1
        
        # SRL
        if read_srl:
            
            # read the roles for each predicate
            for i, role in enumerate(data[ConllPos.pred + 1:]):
                role, expected_role[i] = read_role(role, expected_role[i])
                
                # repeat the tag if the word was splitted
                for _ in range(num_parts):
                    instances[i].append(role)
        
    assert instances is None
    
    return sentences

def read_role(role, expected_role):
    """
    Reads the next semantic role from a CoNLL-style file.
    :return a tuple (role, expected next role)
    """
    if role == '*':
        # signals continuation of the last block
        role = expected_role
    elif role == '*)':
        # finishes block
        role = expected_role
        expected_role = 'O'
    else:
        # verifies if it is a single argument
        match = re.search('\(([-\w]+)\*\)', role)
        if match:
            role = match.group(1)
            expected_role = 'O'
        else:
            # verifies if it opens an argument
            match = re.search('\(([-\w]+)\*', role)
            if match:
                role = match.group(1)
                expected_role = role
            else:
                raise ValueError('Unexpected role data: %s' % role)
    
    if role.startswith('C-'):
        # removes C-
        role = role[2:]
        
    return (role, expected_role)
