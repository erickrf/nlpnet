# -*- coding: utf-8 -*-

"""
Taggers wrapping the neural networks.
"""

import logging
import numpy as np
from itertools import izip

import utils
import config
import attributes
from metadata import Metadata
from pos.macmorphoreader import MacMorphoReader
from srl.srl_reader import SRLReader
from network import Network, ConvolutionalNetwork, LanguageModel

def load_network(md):
    """
    Loads the network from the default file and returns it.
    """
    logger = logging.getLogger("Logger")
    is_srl = md.task.startswith('srl') and md.task != 'srl_predicates'
    
    logger.info('Loading network')
    if is_srl:
        net_class = ConvolutionalNetwork
    elif md.task == 'lm':
        net_class = LanguageModel
    else:
        net_class = Network
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


def create_reader(md, gold_file=None):
    """
    Creates a TextReader object for the given task and loads its dictionary.
    """
    logger = logging.getLogger('Logger')
    logger.info('Loading text reader...')
    
    if md.task == 'pos':
        tr = MacMorphoReader(filename=gold_file)
        tr.load_tag_dict()
        
    elif md.task.startswith('srl'):
        tr = SRLReader(filename=gold_file, only_boundaries= (md.task == 'srl_boundary'),
                       only_classify= (md.task == 'srl_classify'), 
                       only_predicates= (md.task == 'srl_predicates'))
        tr.load_tag_dict()
            
    else:
        raise ValueError("Unknown task: %s" % md.task)
    
    tr.load_dictionary()
    tr.create_converter(md)
    
    logger.info('Done')
    return tr


def _join_2_steps(boundaries, arguments):
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

class Tagger(object):
    """
    Base class for taggers. It should not be instantiated.
    """
    
    def __init__(self):
        """Creates a tagger and loads data preemptively"""
        assert config.data_dir is not None, 'nlpnet data dir is not set.'
        
        self._load_data()
    
    def _load_data(self):
        """Implemented by subclasses"""
        pass

class SRLTagger(Tagger):
    """
    An SRLTagger loads the models and performs SRL on text.
    
    It works on three stages: predicate identification, argument detection and
    argument classification.    
    """
    
    def _load_data(self):
        """Loads data for SRL"""
        # load boundary identification network and reader 
        md_boundary = Metadata.load_from_file('srl_boundary')
        self.boundary_nn = load_network(md_boundary)
        self.boundary_reader = create_reader(md_boundary)
        self.boundary_itd = self.boundary_reader.get_inverse_tag_dictionary()
        
        # same for arg classification
        md_classify = Metadata.load_from_file('srl_classify')
        self.classify_nn = load_network(md_classify)
        self.classify_reader = create_reader(md_classify)
        self.classify_itd = self.classify_reader.get_inverse_tag_dictionary()
        
        # predicate detection
        md_pred = Metadata.load_from_file('srl_predicates')
        self.pred_nn = load_network(md_pred)
        self.pred_reader = create_reader(md_pred)
    
    def find_predicates(self, tokens):
        """
        Finds out which tokens are predicates.
        
        :param tokens: a list of attribute.Token elements
        :returns: the indices of predicate tokens
        """
        sent_codified = np.array([self.pred_reader.converter.convert(token) 
                                  for token in tokens])
        answer = np.array(self.pred_nn.tag_sentence(sent_codified))
        return answer.nonzero()[0]

    def tag(self, tokens, no_repeats=False):
        """
        Runs the SRL process on the given tokens.
        
        :param tokens: a list of tokens (as strings)
        :param no_repeats: whether to prevent repeated argument labels
        :returns: a tuple with the predicates and the tags for each token
        """
        tokens_obj = [attributes.Token(t) for t in tokens]
        converted_bound = np.array([self.boundary_reader.converter.convert(t) 
                                    for t in tokens_obj])
        converted_class = np.array([self.classify_reader.converter.convert(t) 
                                    for t in tokens_obj])
        
        pred_positions = self.find_predicates(tokens_obj)
        preds = [t if i in pred_positions else '-' for i, t in enumerate(tokens)]
        
        # first, argument boundary detection
        # the answer includes all predicates
        answers = self.boundary_nn.tag_sentence(converted_bound, pred_positions)
        boundaries = [[self.boundary_itd[x] for x in pred_answer] 
                      for pred_answer in answers]
        arg_limits = [utils.boundaries_to_arg_limits(pred_boundaries) 
                      for pred_boundaries in boundaries]
        
        # now, argument classification
        answers = self.classify_nn.tag_sentence(converted_class, 
                                                pred_positions, arg_limits,
                                                allow_repeats=not no_repeats)
        arguments = [[self.classify_itd[x] for x in pred_answer] 
                     for pred_answer in answers]
        
        joined = _join_2_steps(boundaries, arguments)
        return ((preds, joined))


class POSTagger(Tagger):
    """A POSTagger loads the models and performs POS tagging on text."""
    
    def _load_data(self):
        """Loads data for POS"""
        md = Metadata.load_from_file('pos')
        self.nn = load_network(md)
        self.reader = create_reader(md)
        self.itd = self.reader.get_inverse_tag_dictionary()
    
    def tag(self, tokens):
        """
        Tags the given tokens for part-of-speech.
        
        :param tokens: a list of tokens (as strings)
        """
        converted_tokens = np.array([self.reader.converter.convert(token) 
                                     for token in tokens])
        answer = self.nn.tag_sentence(converted_tokens)
        tags = [self.itd[tag] for tag in answer]
        return tags
    

            