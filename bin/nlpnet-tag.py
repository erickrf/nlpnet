#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script will run a POS or SRL tagger on the input data and print the results
to stdout.
"""

import argparse
import logging
from itertools import izip

import nlpnet.taggers as taggers
import nlpnet.utils as utils

def interactive_running(task):
    """
    This function provides an interactive environment for running the system.
    It receives text from the standard input, tokenizes it, and calls the function
    given as a parameter to produce an answer.
    
    :param task: either 'pos' or 'srl'
    """
    task_lower = task.lower()
    if task_lower == 'pos':
        tagger = taggers.POSTagger()
    elif task_lower == 'srl':
        tagger = taggers.SRLTagger()
    else:
        raise ValueError('Unknown task: %s' % task)
    
    while True:
        try:
            text = raw_input()
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        
        if type(text) is not unicode:
            text = unicode(text, 'utf-8')
        
        tokens = utils.tokenize(text, wiki=False)
        result = []
        for sent in tokens:
            # we use the cleaned tokens to match the entries in the dictionary, 
            # and the original (not cleaned) tokens to return in the answer. 
            sent_cleaned = [utils.clean_text(token, correct=False) for token in sent]
            tags = tagger.tag(sent_cleaned)
            result.append((sent, tags))
            
        _print_tags(result)

def _print_tags(data):
    """
    Prints text with its corresponding tags.
    
    :param data: a list of (sentence, tags) tuples. A sentence is a list of tokens,
    and the tags depend on the task. In POS, they are a simple list, and in SRL, 
    they are a tuple (predicates, tags), where there is one tag list for each 
    predicate. 
    """
    # each item in the data corresponds to a sentence
    for sent in data:
        actual_sent, tags = sent
        # get the length of the longer token to have a nice formatting
        max_len_token = max(len(token) for token in actual_sent)
        format_str = u'{:<%d}' % (max_len_token + 1)
        actual_sent = [format_str.format(token) for token in actual_sent]
        
        # in srl, the first element of the tags contains the verbs
        if isinstance(tags, tuple):
            verbs, srl_tags = tags
            max_len_verb = max(len(token) for token in verbs)
            format_str = u'{:<%d}' % (max_len_verb + 1)
            verbs = [format_str.format(verb) for verb in verbs]
            
            # the asterisk tells izip to treat the elements in the list as separate arguments
            the_iter = izip(actual_sent, verbs, *srl_tags)
        else:
            the_iter = izip(actual_sent, tags)
        
        for token_and_tags in the_iter:
            print '\t'.join(token_and_tags).encode('utf-8')
            
        # linebreak after each sentence
        print


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task for which the network should be used.', 
                        type=str, required=True, choices=['srl', 'pos'])
    parser.add_argument('-v', help='Verbose mode', action='store_true', dest='verbose')
    parser.add_argument('--no-repeat', dest='no_repeat', action='store_true',
                        help='Forces the classification step to avoid repeated argument labels (SRL only).')
    args = parser.parse_args()
    
    logging_level = logging.DEBUG if args.verbose else logging.WARNING
    utils.set_logger(logging_level)
    logger = logging.getLogger("Logger")
    
    interactive_running(args.task)
    