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
from pos.pos_reader import POSReader
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
    
    logger.info('Loading features...')
    type_features = utils.load_features_from_file(config.FILES[md.type_features])
    tables = [type_features]
    
    if md.use_caps:
        caps_features = utils.load_features_from_file(config.FILES[md.caps_features])
        tables.append(caps_features)
    if md.use_prefix:
        prefix_features = utils.load_features_from_file(config.FILES[md.prefix_features])
        for table in prefix_features:
            # one table for each size
            tables.append(table)
    if md.use_suffix:
        suffix_features = utils.load_features_from_file(config.FILES[md.suffix_features])
        for table in suffix_features:
            # one table for each size
            tables.append(table)
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
    :param md: a metadata object describing the task
    :param gold_file: path to a file with gold standard data, if
        the reader will be used for testing.
    """
    logger = logging.getLogger('Logger')
    logger.info('Loading text reader...')
    
    if md.task == 'pos':
        tr = POSReader(filename=gold_file)
        
    elif md.task.startswith('srl'):
        tr = SRLReader(filename=gold_file, only_boundaries= (md.task == 'srl_boundary'),
                       only_classify= (md.task == 'srl_classify'), 
                       only_predicates= (md.task == 'srl_predicates'))
            
    else:
        raise ValueError("Unknown task: %s" % md.task)
    
    tr.create_converter(md)
    
    logger.info('Done')
    return tr

def _group_arguments(tokens, predicate_positions, boundaries, labels):
    """
    Groups words pertaining to each argument and returns a dictionary for each predicate.
    """
    arg_structs = []
    
    for predicate_position, pred_boundaries, pred_labels in izip(predicate_positions,
                                                                 boundaries, 
                                                                 labels):
        structure = {}
        
        for token, boundary_tag in izip(tokens, pred_boundaries):
            if boundary_tag == 'O':
                continue
            
            elif boundary_tag == 'B':
                argument_tokens = [token]            
            
            elif boundary_tag == 'I':
                argument_tokens.append(token)  
                
            elif boundary_tag == 'E': 
                argument_tokens.append(token)
                tag = pred_labels.pop(0)
                structure[tag] = argument_tokens
            
            else:
                # boundary_tag == 'S'
                tag = pred_labels.pop(0)
                structure[tag] = [token]
        
        predicate = tokens[predicate_position]
        arg_structs.append((predicate, structure))
    
    return arg_structs
        

class SRLAnnotatedSentence(object):
    """
    Class storing a sentence with annotated semantic roles.
    
    It stores a list with the sentence tokens, called `tokens`, and a list of tuples
    in the format `(predicate, arg_strucutres)`. Each `arg_structure` is a dict mapping 
    semantic roles to the words that constitute it. This is used instead of a two-level
    dictionary because one sentence may have more than one occurrence of the same 
    predicate.
    
    This class is used only for storing data.
    """
    
    def __init__(self, tokens, arg_structures):
        """
        Creates an instance of a sentence with SRL data.
        
        :param tokens: a list of strings
        :param arg_structures: a list of tuples in the format (predicate, mapping).
            Each predicate is a string and each mapping is a dictionary mapping role labels
            to the words that constitute it. 
        """
        self.tokens = tokens
        self.arg_structures = arg_structures
        


class Tagger(object):
    """
    Base class for taggers. It should not be instantiated.
    """
    
    def __init__(self, tokenizer=None):
        """Creates a tagger and loads data preemptively"""
        asrt_msg = "nlpnet data directory is not set. \
If you don't have the trained models, download them from http://nilc.icmc.usp.br/nilc/download/nlpnet-data.zip"
        assert config.data_dir is not None, asrt_msg
        
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

    def tag(self, text, no_repeats=False):
        """
        Runs the SRL process on the given text.
        
        :param text: unicode or str encoded in utf-8.
        :param no_repeats: whether to prevent repeated argument labels
        :returns: a list of SRLAnnotatedSentence objects
        """
        tokens = utils.tokenize(text, clean=False)
        result = []
        for sent in tokens:
            tagged = self.tag_tokens(sent)
            result.append(tagged)
        
        return result

    def tag_tokens(self, tokens, no_repeats=False):
        """
        Runs the SRL process on the given tokens.
        
        :param tokens: a list of tokens (as strings)
        :param no_repeats: whether to prevent repeated argument labels
        :returns: a list of lists (one list for each sentence). Sentences have tuples 
            (all_tokens, predicate, arg_structure), where arg_structure is a dictionary 
            mapping argument labels to the words it includes.
        """
        tokens_obj = [attributes.Token(utils.clean_text(t, False)) for t in tokens]
        converted_bound = np.array([self.boundary_reader.converter.convert(t) 
                                    for t in tokens_obj])
        converted_class = np.array([self.classify_reader.converter.convert(t) 
                                    for t in tokens_obj])
        
        pred_positions = self.find_predicates(tokens_obj)
        
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
        
        structures = _group_arguments(tokens, pred_positions, boundaries, arguments)
        return SRLAnnotatedSentence(tokens, structures)
        

class POSTagger(Tagger):
    """A POSTagger loads the models and performs POS tagging on text."""
    
    def _load_data(self):
        """Loads data for POS"""
        md = Metadata.load_from_file('pos')
        self.nn = load_network(md)
        self.reader = create_reader(md)
        self.itd = self.reader.get_inverse_tag_dictionary()
    
    def tag(self, text):
        """
        Tags the given text.
        
        :param text: a string or unicode object. Strings assumed to be utf-8
        :returns: a list of lists (sentences with tokens). 
            Each sentence has (token, tag) tuples.
        """
        tokens = utils.tokenize(text, clean=False)
        result = []
        for sent in tokens:
            tagged = self.tag_tokens(sent, return_tokens=True)
            result.append(tagged)
        
        return result
    
    def tag_tokens(self, tokens, return_tokens=False):
        """
        Tags a given list of tokens. 
        
        Tokens should be produced with the nlpnet tokenizer in order to 
        match the entries in the vocabulary. If you have non-tokenized text,
        use POSTagger.tag(text).
        
        :param tokens: a list of strings
        :param return_tokens: if True, includes the tokens in the return,
            as a list of tuples (token, tag).
        :returns: a list of strings (the tags)
        """
        converter = self.reader.converter
        converted_tokens = np.array([converter.convert(utils.clean_text(token, False)) 
                                     for token in tokens])
        answer = self.nn.tag_sentence(converted_tokens)
        tags = [self.itd[tag] for tag in answer]
        
        if return_tokens:
            return zip(tokens, tags)
        
        return tags

