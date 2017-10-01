# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Taggers wrapping the neural networks.
"""

import logging
import numpy as np
from six.moves import zip

from nlpnet import utils
from nlpnet import config
from nlpnet import attributes
from nlpnet.metadata import Metadata
from nlpnet.pos import POSReader
from nlpnet.srl import SRLReader
from nlpnet.parse import DependencyReader
from nlpnet.network import Network, ConvolutionalNetwork, \
    ConvolutionalDependencyNetwork


def load_network(md):
    """
    Loads the network from the default file and returns it.
    """
    logger = logging.getLogger("Logger")
    is_srl = md.task.startswith('srl') and md.task != 'srl_predicates'
    
    logger.info('Loading network')
    if is_srl:
        net_class = ConvolutionalNetwork
    elif md.task.endswith('dependency'):
        net_class = ConvolutionalDependencyNetwork
    else:
        net_class = Network
    nn = net_class.load_from_file(md.paths[md.network])
    
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
        tr = POSReader(md, filename=gold_file)
    
    elif 'dependency' in md.task:
        labeled = md.task.startswith('labeled')
        tr = DependencyReader(md, filename=gold_file, labeled=labeled)
    
    elif md.task.startswith('srl'):
        tr = SRLReader(md, filename=gold_file,
                       only_boundaries= (md.task == 'srl_boundary'),
                       only_classify= (md.task == 'srl_classify'),
                       only_predicates= (md.task == 'srl_predicates'))
    
    else:
        raise ValueError("Unknown task: %s" % md.task)
    
    logger.info('Done')
    return tr


def _group_arguments(tokens, predicate_positions, boundaries, labels):
    """
    Groups words pertaining to each argument and returns a dictionary for
    each predicate.
    """
    arg_structs = []
    
    for item in zip(predicate_positions, boundaries, labels):
        predicate_position, pred_boundaries, pred_labels = item
        structure = {}

        for token, boundary_tag in zip(tokens, pred_boundaries):
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

    It stores a list with the sentence tokens, called `tokens`, and a list of
    tuples in the format `(predicate, arg_strucutres)`. Each `arg_structure` is
    a dict mapping semantic roles to the words that constitute it. This is used
    instead of a two-level dictionary because one sentence may have more than
    one occurrence of the same predicate.

    This class is used only for storing data.
    """

    def __init__(self, tokens, arg_structures):
        """
        Creates an instance of a sentence with SRL data.

        :param tokens: a list of strings
        :param arg_structures: a list of tuples in the format (predicate,
            mapping).
            Each predicate is a string and each mapping is a dictionary mapping
            role labels to the words that constitute it.
        """
        self.tokens = tokens
        self.arg_structures = arg_structures


class ParsedSentence(object):
    """
    Class for storing a sentence with dependency parsing annotation.
    
    It stores a list of tokens, the dependency heads, dependency labels and POS
    tags if the parser used them. Dependency heads are the index of the head of
    each token, and -1 means a dependency to the root.
    """
    def __init__(self, tokens, heads, labels, pos=None):
        """
        Constructor.
        
        :param tokens: list of strings
        :param heads: list of integers (-1 means dependency to root, others
            are token indices)
        :param labels: list of strings
        :param pos: None or list of strings
        """
        self.tokens = tokens
        self.heads = heads
        self.labels = labels
        self.pos = pos
    
    def __len__(self):
        return len(self.tokens)
    
    def to_conll(self):
        """
        Return a string representation of the sentence in CoNLL X format. 
        
        Each line has:
        [number starting from 1] token _ POS POS _ head label
        
        Token numbers start from 1, root is referred as 0.
        POS is only available if the original parser used it.
        """
        result = []
        for i in range(len(self.tokens)):
            token = self.tokens[i]
            head = self.heads[i] + 1
            label = self.labels[i]
            pos = self.pos[i] if self.pos else '_'
            
            line = '{id}\t{token}\t_\t{pos}\t{pos}\t_\t{head}\t{label}'
            result.append(line.format(id=i+1, pos=pos, head=head, label=label,
                                      token=token))
        
        return '\n'.join(result)


class Tagger(object):
    """
    Base class for taggers. It should not be instantiated.
    """
    def __init__(self, data_dir=None, language='en'):
        """Creates a tagger and loads data preemptively"""
        asrt_msg = "nlpnet data directory is not set. If you don't have the " \
                   "trained models, download them from " \
                   "http://nilc.icmc.usp.br/nlpnet/models.html"
        if data_dir is None:
            assert config.data_dir is not None, asrt_msg
            self.paths = config.FILES
        else:
            self.paths = config.get_config_paths(data_dir)
        
        self.data_dir = data_dir
        self.language = language
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
        md_boundary = Metadata.load_from_file('srl_boundary', self.paths)
        self.boundary_nn = load_network(md_boundary)
        self.boundary_reader = create_reader(md_boundary)
        self.boundary_reader.create_converter()
        self.boundary_itd = self.boundary_reader.get_inverse_tag_dictionary()
        
        # same for arg classification
        md_classify = Metadata.load_from_file('srl_classify', self.paths)
        self.classify_nn = load_network(md_classify)
        self.classify_reader = create_reader(md_classify)
        self.classify_reader.create_converter()
        self.classify_itd = self.classify_reader.get_inverse_tag_dictionary()
        
        # predicate detection
        md_pred = Metadata.load_from_file('srl_predicates', self.paths)
        self.pred_nn = load_network(md_pred)
        self.pred_reader = create_reader(md_pred)
        self.pred_reader.create_converter()
    
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
    
    def tag(self, text):
        """
        Runs the SRL process on the given text.

        :param text: unicode or str encoded in utf-8.
        :returns: a list of SRLAnnotatedSentence objects
        """
        tokens = utils.tokenize(text, self.language)
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
        :returns: a list of lists (one list for each sentence).
            Sentences have tuples (all_tokens, predicate, arg_structure), where
            arg_structure is a dictionary mapping argument labels to the words
            it includes.
        """
        if self.language == 'pt':
            tokens_obj = [attributes.Token(utils.clean_text(t, False))
                          for t in tokens]
        else:
            tokens_obj = [attributes.Token(t) for t in tokens]
        
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
        
        structures = _group_arguments(tokens, pred_positions, boundaries,
                                      arguments)
        return SRLAnnotatedSentence(tokens, structures)


class DependencyParser(Tagger):
    """A Dependency Parser based on a neural network tagger."""
    
    def __init__(self, *args, **kwargs):
        """
        Set the data directory for the POS tagger, if one is used,
        and call the parent constructor.        
        """
        super(DependencyParser, self).__init__(*args, **kwargs)
    
    def _load_data(self):
        """Loads data for Dependency Parsing"""
        md_udep = Metadata.load_from_file('unlabeled_dependency',
                                          paths=self.paths)
        self.unlabeled_nn = load_network(md_udep)
        self.unlabeled_reader = create_reader(md_udep)
        
        md_ldep = Metadata.load_from_file('labeled_dependency',
                                          paths=self.paths)
        self.labeled_nn = load_network(md_ldep)
        self.labeled_reader = create_reader(md_ldep)
        self.itd = self.labeled_reader.get_inverse_tag_dictionary()
        
        self.use_pos = md_udep.use_pos or md_ldep.use_pos
        if self.use_pos:
            self.pos_tagger = POSTagger(self.data_dir, language=self.language)
    
    def parse(self, text):
        """
        Split the given text into sentences and determines their 
        dependency trees. If you want to provide your own tokenized
        text, use `parse_sentence` instead.
                
        :param text: a string
        :returns: a list of ParsedSentence's
        """
        sentences = utils.tokenize(text, self.language)
        result = []
        for sent in sentences:
            parsed = self.parse_sentence(sent)
            result.append(parsed)
        
        return result
    
    def tag_tokens(self, tokens):
        """
        Parse the given sentence. This function is just an alias for
        `parse_sentence`.
        """
        return self.parse_sentence(tokens)
    
    def parse_sentence(self, tokens):
        """
        Parse the given sentence. It must be already tokenized; if you
        want nlpnet to tokenize the text, use the method `parse` instead.
        
        :param tokens: a list of strings
        :return: a ParsedSentence instance
        """
        original_tokens = tokens
        tokens_obj = []
        
        # if the parser uses POS a feature, have a tagger tag it first
        if self.use_pos:
            tokens = self.pos_tagger.tag_tokens(tokens, return_tokens=True)
        
        for token in tokens:
            if self.use_pos:
                # if we tagged for POS, each item is a tuple
                word, pos = token
            else:
                pos = None
            tokens_obj.append(attributes.Token(word, pos=pos))
        
        converted_tokens = self.unlabeled_reader.codify_sentence(tokens_obj)
        heads = self.unlabeled_nn.tag_sentence(converted_tokens)
        
        # the root is returned having a value == len(sentence)
        root = heads.argmax()
        heads[root] = root
        
        converted_tokens = self.labeled_reader.codify_sentence(tokens_obj)
        label_codes = self.labeled_nn.tag_sentence(converted_tokens, heads)
        labels = [self.itd[code] for code in label_codes]
        
        # to the final answer, signal the root with -1
        heads[root] = -1
        if self.use_pos:
            # unzip
            pos_tags = zip(*tokens)[1]
        else:
            pos_tags = None
            
        parsed = ParsedSentence(original_tokens, heads, labels, pos_tags)
        return parsed
    
    def tag(self, text):
        """
        Parse the given text. This is just an alias for the 
        `parse` method.
        """
        return self.parse(text)


class POSTagger(Tagger):
    """A POSTagger loads the models and performs POS tagging on text."""

    def _load_data(self):
        """Loads data for POS"""
        md = Metadata.load_from_file('pos', self.paths)
        self.nn = load_network(md)
        self.reader = create_reader(md)
        self.reader.create_converter()
        self.itd = self.reader.get_inverse_tag_dictionary()

    def tag(self, text):
        """
        Tags the given text.

        :param text: a string or unicode object. Strings assumed to be utf-8
        :returns: a list of lists (sentences with tokens).
            Each sentence has (token, tag) tuples.
        """
        tokens = utils.tokenize(text, self.language)
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
        converted_tokens = np.array([converter.convert(token) 
                                     for token in tokens])
            
        answer = self.nn.tag_sentence(converted_tokens)
        tags = [self.itd[tag] for tag in answer]

        if return_tokens:
            return list(zip(tokens, tags))

        return tags

