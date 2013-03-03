# -*- coding: utf-8 -*-

"""
This script will run the trained models for NLP tasks.
"""

import argparse
import logging
import numpy as np
from itertools import izip

import utils
import config
import attributes
from metadata import Metadata
from pos.macmorphoreader import MacMorphoReader
from srl.srl_reader import SRLReader
from nlpnet import Network, ConvolutionalNetwork

def interactive_running(network_caller):
    """
    This function provides an interactive environment for running the system.
    It receives text from the standard input, tokenizes it, and calls the function
    given as a parameter to produce an answer.
    @param network_caller: a function that receives a list of tokens and returns
    the tags by the network.
    """
    while True:
        try:
            text = raw_input()
        except KeyboardInterrupt:
            break
        except EOFError:
            break
        
        if type(text) is not unicode:
            text = unicode(text, 'utf-8')
        
        tagged = tag_text(text, network_caller)
        print_tags(tagged)

def tag_text(text, tagger):
    """
    Tokenizes the text and tags it calling a given function.
    @return: a list containing all sentences and their tags. Each sentence
    is represented by a tuple with two elements: the sentence tokens and the
    tags. 
    """
    # we use the cleaned tokens to match the entries in the dictionary, and the original
    # (not cleaned) tokens to return in the answer. 
    tokens = utils.tokenize(text, wiki=False)
    result = []
    
    for sent in tokens:
        sent_cleaned = [utils.clean_text(token, correct=False) for token in sent]
        tags = tagger(sent_cleaned)
        
        result.append((sent, tags))
    
    return result


def print_tags(data):
    """
    Prints text with its corresponding tags, as returned by the networks.
    """
    # each item in the data corresponds to a sentence
    for sent in data:
        actual_sent, tags = sent
        
        # in srl, the first element of the tags contains the verbs
        if isinstance(tags, tuple):
            verbs, srl_tags = tags
            
            # the asterisk tells izip to treat the elements in the list as separate arguments
            the_iter = izip(actual_sent, verbs, *srl_tags)
        else:
            the_iter = izip(actual_sent, tags)
        
        for token_and_tags in the_iter:
            
            # have additional space for the token, ir order to keep things aligned
            token = token_and_tags[0]
            token = '{:<20}'.format(token.encode('utf-8'))
            
            tags = '\t'.join(token_and_tags[1:]).encode('utf-8')
            
            print '%s\t%s' % (token, tags)
            
        # linebreak after each sentence
        print

def load_network(md):
    """
    Loads the network from the default file and returns it.
    """
    logger = logging.getLogger("Logger")
    is_srl = md.task.startswith('srl') and md.task != 'srl_predicates'
    
    logger.info('Loading network')
    net_class = ConvolutionalNetwork if is_srl else Network
    nn = net_class.load_from_file(config.FILES[md.network])
    
    logger.info('Loading features')
    type_features = utils.load_features_from_file(config.FILES[md.type_features])
    tables = [type_features]
    
    if md.use_caps:
        caps_features = utils.load_features_from_file(config.FILES[md.caps_features])
        tables.append(caps_features)
    if md.use_suffix:
        suffix_features = utils.load_features_from_file(config.FILES[md.suffix_features])
        tables.append(suffix_features)
    if md.use_pos:
        pos_features = utils.load_features_from_file(config.FILES[md.pos_features])
        tables.append(pos_features)
    if md.use_chunk:
        chunk_features = utils.load_features_from_file(config.FILES[md.chunk_features])
        tables.append(chunk_features)
        
    nn.feature_tables = tables
    
    logger.info('Done')
    return nn


def create_reader_pos(gold=False):
    """
    Creates a TextReader object ready for POS.
    """
    if gold:
        pos_reader = MacMorphoReader(filename=config.FILES['macmorpho_test'])
    else:
        pos_reader = MacMorphoReader([])
    return pos_reader

def create_reader_srl(gold=False, only_boundaries=False, only_classify=False,
                      only_predicates=False):
    """
    Creates a simple TextReader object ready for SRL.
    """
    if gold:
        reader = SRLReader(filename=config.FILES['conll_test'], only_boundaries=only_boundaries,
                           only_classify=only_classify, only_predicates=only_predicates)
    else:
        reader = SRLReader([[], []], only_boundaries=only_boundaries, only_classify=only_classify,
                           only_predicates=only_predicates)
        
    return reader
    

def create_reader(md, gold):
    """
    Creates a TextReader object for the given task and loads its dictionary.
    """
    logger = logging.getLogger('Logger')
    logger.info('Loading text reader...')
    
    if md.task == 'pos':
        tr = create_reader_pos(gold)
        tr.load_tag_dict()
    elif md.task.startswith('srl'):
        tr = create_reader_srl(gold, md.task == 'srl_boundary', md.task == 'srl_classify',
                               md.task == 'srl_predicates')
        tr.load_tag_dict()
    else:
        raise ValueError("Unknown task: %s" % md.task)
    
    tr.load_dictionary()
    tr.create_converter(md)
    
    logger.info('Done')
    return tr

def run_2_steps(nn_boundary, nn_classify, sent_boundary, sent_classify,
                itd_boundary, itd_classify, predicates, no_repeat=False):
    """
    Runs the 2-step system for a given sentence. The sentence must be provided
    with the boundary and classification codification separately, as they
    can be different.
    @param nn_boundary: argument recognition network
    @param nn_classify: argument classification network
    @param sent_boundary: sentence with tokens codified for recognition
    @param sent_classify: sentence with tokens codified for classification
    @param itd_boundary: inverse tag dictionary for boundaries (maps numbers to names)
    @param itd_classify: inverse tag dictionary for classification
    @param predicates: the positions of the predicates in the sentence
    @return: a list of IOBES arguments
    """
    # the answer includes all predicates
    answers = nn_boundary.tag_sentence(sent_boundary, predicates)
    boundaries = [[itd_boundary[x] for x in pred_answer] for pred_answer in answers]
    
    arg_limits = [utils.boundaries_to_arg_limits(pred_boundaries) 
                  for pred_boundaries in boundaries]
    
    answers = nn_classify.tag_sentence(sent_classify, 
                                       predicates, arg_limits,
                                       allow_repeats=not no_repeat)
    
    arguments = [[itd_classify[x] for x in pred_answer] for pred_answer in answers]

    return join_2_steps(boundaries, arguments)


def join_2_steps(boundaries, arguments):
    """
    Joins the tags for argument boundaries and classification accordingly.
    """
    answer = []
    
    for pred_boundaries, pred_arguments in izip(boundaries, arguments):
        cur_arg = ''
        pred_answer = []
        
        for boundary_tag in pred_boundaries:
            if boundary_tag == 'O':
                pred_answer.append('O')
            elif boundary_tag in 'BS':
                cur_arg = pred_arguments.pop(0)
                tag = '%s-%s' % (boundary_tag, cur_arg)
                pred_answer.append(tag)
            else:
                tag = '%s-%s' % (boundary_tag, cur_arg)
                pred_answer.append(tag)
        
        answer.append(pred_answer)
    
    return answer

def get_predicate_finder():
    """
    Returns a function to find the predicates in a sentence.
    """
    # load the predicate finder network and reader
    md = Metadata.load_from_file('srl_predicates')
    nn = load_network(md)
    reader = create_reader(md, gold=False)
    
    def predicate_finder(sentence):
        sent_codified = np.array([reader.converter.convert(token) for token in sentence])
        answer = np.array(nn.tag_sentence(sent_codified))
        return answer.nonzero()[0]
    
    return predicate_finder


def tag_srl(no_repeats=False):
    """
    Runs the SRL process in two steps. First, a network will identify
    argument boundaries, then, another one will classify them.
    """
    # load boundary identification network and reader 
    md_boundary = Metadata.load_from_file('srl_boundary')
    nn_boundary = load_network(md_boundary)
    reader_boundary = create_reader(md_boundary, False)
    itd_boundary = reader_boundary.get_inverse_tag_dictionary()
    
    # same for arg classification
    md_classify = Metadata.load_from_file('srl_classify')
    nn_classify = load_network(md_classify)
    reader_classify = create_reader(md_classify, False)
    itd_classify = reader_classify.get_inverse_tag_dictionary()
    
    pred_finder = get_predicate_finder()
    
    # function to be passed to the interactive loop
    def srl_2_steps_(tokens):
        tokens_obj = [attributes.Token(t) for t in tokens]
        converted_bound = np.array([reader_boundary.converter.convert(t) for t in tokens_obj])
        converted_class = np.array([reader_classify.converter.convert(t) for t in tokens_obj])
        
        pred_pos = pred_finder(tokens_obj)
        preds = [t if i in pred_pos else '-' for i, t in enumerate(tokens)]
        
        srl_tags = run_2_steps(nn_boundary, nn_classify, converted_bound, converted_class, 
                               itd_boundary, itd_classify, pred_pos, no_repeats)
        
        return ((preds, srl_tags))
        
    interactive_running(srl_2_steps_)

def tag_pos(heuristics=False, interactive=True):
    """
    Runs the POS network.
    """
    # load the POS network and reader
    md = Metadata.load_from_file('pos')
    nn = load_network(md)
    reader = create_reader(md, gold=False)
    
    itd = reader.get_inverse_tag_dictionary()
    
    # function for the interactive loop
    def tag_pos_(tokens):
        tokens = np.array([reader.converter.convert(token) for token in tokens])
        answer = nn.tag_sentence(tokens)
        tags = [itd[tag] for tag in answer]
        return tags
    
    if interactive:
        interactive_running(tag_pos_)
    else:
        return tag_pos_

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='Task for which the network should be used.', 
                        type=str, default='pos', choices=['srl', 'pos'])
    parser.add_argument('-v', help='Verbose mode', action='store_true', dest='verbose')
    parser.add_argument('--no-repeat', dest='no_repeat', action='store_true',
                        help='Forces the classification step to avoid repeated argument labels (SRL only).')
    args = parser.parse_args()
    
    logging_level = logging.DEBUG if args.verbose else logging.WARNING
    utils.set_logger(logging_level)
    logger = logging.getLogger("Logger")
    
    if args.task == 'pos':
        tag_pos()
    elif args.task.startswith('srl'):
        tag_srl(args.no_repeat)
        