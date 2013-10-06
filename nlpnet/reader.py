#!/usr/env python
# -*- coding: utf-8 -*-

"""
Base class for reading NLP tagging data.
"""

import cPickle
import logging
import numpy as np
from collections import Counter

import config
from word_dictionary import WordDictionary
import attributes
from attributes import get_capitalization

class TextReader(object):
    
    def __init__(self, sentences=None, filename=None):
        """
        :param sentences: A list of lists of tokens.
        :param filename: Alternatively, the name of the file from where sentences 
        can be read. The file should have one sentence per line, with tokens
        separated by white spaces.
        """
        if sentences is not None:
            self.sentences = sentences
        else:
            self.sentences = []
            with open(filename, 'rb') as f:
                for line in f:
                    sentence = unicode(line, 'utf-8').split()
                    self.sentences.append(sentence)
                    
        self.converter = None
        self.task = 'lm'
    
    def add_text(self, text):
        """
        Adds more text to the reader. The text must be a sequence of sequences of 
        tokens.
        """
        self.sentences.extend(text)
    
    def load_dictionary(self, filename=config.FILES['word_dict_dat']):
        """
        Reads a dictionary from a pickled file.
        """
        logger = logging.getLogger("Logger")
        logger.info("Loading provided dictionary...")
        with open(filename, 'rb') as f:
            wd = cPickle.load(f)
        wd.check()
        self.word_dict = wd
        logger.info("Done. Dictionary size is %d types" % wd.num_tokens)
    
    def generate_dictionary(self, dict_size=None, minimum_occurrences=None):
        """
        Generates a token dictionary based on the supplied text.
        :param dict_size: Max number of tokens to be included in the dictionary.
        :param minimum_occurrences: Minimum number of times that a token must
        appear in the text in order to be included in the dictionary. 
        """
        logger = logging.getLogger("Logger")
        logger.info("Creating dictionary...")
        
        self.word_dict = WordDictionary(self.sentences, dict_size, minimum_occurrences)
            
        logger.info("Done. Dictionary size is %d tokens" % self.word_dict.num_tokens)
    
    def save_word_dict(self):
        """
        Saves the reader's word dictionary in cPickle format.
        """
        logger = logging.getLogger("Logger")
        
        with open(config.FILES['word_dict_dat'], 'wb') as f:
            cPickle.dump(self.word_dict, f, 2)
            
        logger.info("Dictionary saved in %s" % config.FILES['word_dict_dat'])
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        """
        new_sentences = []
        for sent in self.sentences:
            new_sent = np.array([self.converter.convert(token) for token in sent])
            new_sentences.append(new_sent)
        
        self.sentences = new_sentences
    
    def create_converter(self, metadata):
        """
        Sets up the token converter, which is responsible for transforming tokens into their
        feature vector indices
        """
        self.converter = attributes.TokenConverter()
        self.converter.add_extractor(self.word_dict.get)
        if metadata.use_caps:
            self.converter.add_extractor(get_capitalization)
        if metadata.use_suffix:
            attributes.Suffix.load_suffixes()
            self.converter.add_extractor(attributes.Suffix.get_suffix)
    

class TaggerReader(TextReader):
    """
    Abstract class extending TextReader with useful functions
    for tagging tasks. 
    """
    
    def __init__(self):
        raise NotImplemented('Abstract class')
    
    def get_inverse_tag_dictionary(self):
        """
        Returns a version of the tag dictionary that maps numbers to tags.
        Used for consulting the meaning of the network's output.
        """
        tuples = [(x[1], x[0]) for x in self.tag_dict.iteritems()]
        ret = dict(tuples)
        
        return ret
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        """
        new_sentences = []
        self.tags = []
        rare_tag_value = self.tag_dict.get(self.rare_tag)
        
        for sent in self.sentences:
            new_sent = []
            sentence_tags = []
            
            for token, tag in sent:
                new_token = self.converter.convert(token)
                new_sent.append(new_token)
                sentence_tags.append(self.tag_dict.get(tag, rare_tag_value))
            
            new_sentences.append(np.array(new_sent))
            self.tags.append(np.array(sentence_tags))
        
        self.sentences = new_sentences
        self.codified = True
    
    def get_word_counter(self):
        """
        Returns a Counter object with word type occurrences.
        """
        c = Counter(token.lower() for sent in self.sentences for token, _ in sent)
        return c
    
    def get_tag_counter(self):
        """
        Returns a Counter object with tag occurrences.
        """
        c = Counter(tag for sent in self.sentences for _, tag in sent)
        return c
    
    def save_tag_dict(self):
        """
        Saves the tag dictionary to a file.
        """
        key = '%s_tag_dict' % self.task
        with open(config.FILES[key], 'w') as f:
            cPickle.dump(self.tag_dict, f)
    
    def load_tag_dict(self, filename=None):
        """
        Loads the tag dictionary from the default file.
        """
        if filename is None:
            key = '%s_tag_dict' % self.task
            filename = config.FILES[key]
            
        with open(filename, 'rb') as f:
            self.tag_dict = cPickle.load(f)
    
    
    