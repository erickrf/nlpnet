#!/usr/env python
# -*- coding: utf-8 -*-

"""
Base class for reading NLP tagging data.
"""

import os
import logging
import numpy as np
from collections import Counter

import config
import attributes
from word_dictionary import WordDictionary
from attributes import get_capitalization

def load_tag_dict(filename):
    """
    Load a tag dictionary from a file containing one tag
    per line.
    """
    tag_dict = {}
    with open(filename, 'rb') as f:
        code = 0
        for tag in enumerate(f):
            tag = unicode(tag, 'utf-8').strip()
            if tag:
                tag_dict[tag] = code
                code += 1
    
    return tag_dict

def save_tag_dict(tag_dict, filename):
    """
    Save the given tag dictionary to the given file. Dictionary
    is saved with one tag per line, in the order of their codes.
    """
    ordered_keys = sorted(tag_dict, key=tag_dict.get)
    text = '\n'.join(ordered_keys)
    with open(filename, 'wb') as f:
        f.write(text.encode('utf-8'))
    

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
    
    def load_dictionary(self):
        """Read a file with a word list and create a dictionary."""
        logger = logging.getLogger("Logger")
        logger.info("Loading vocabulary")
        filename = config.FILES['vocabulary']
        
        words = []
        with open(filename, 'rb') as f:
            for word in f:
                word = unicode(word, 'utf-8').strip()
                if word:
                    words.append(word)
        
        wd = WordDictionary.init_from_wordlist(words)
        self.word_dict = wd
        logger.info("Done. Dictionary size is %d types" % wd.num_tokens)
    
    def generate_dictionary(self, dict_size=None, minimum_occurrences=2):
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
    
    def save_dictionary(self, filename=None):
        """
        Saves the reader's word dictionary as a list of words.
        
        :param filename: path to the file to save the dictionary. 
            if not given, it will be saved in the default nlpnet
            data directory.
        """
        logger = logging.getLogger("Logger")
        if filename is None:
            filename = config.FILES['vocabulary']
        
        with open(filename, 'wb') as f:
            for word in self.word_dict:
                line = '%s\n' % word
                f.write(line.encode('utf-8'))
            
        logger.info("Dictionary saved in %s" % filename)
    
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
        def add_affix_extractors(affix):
            """
            Helper function that works for both suffixes and prefixes.
            The parameter affix should be 'suffix' or 'prefix'.
            """
            loader_function = getattr(attributes.Affix, 'load_%ses' % affix)
            loader_function()
            
            # deal with gaps between sizes (i.e., if there are sizes 2, 3, and 5)
            codes = getattr(attributes.Affix, '%s_codes' % affix)
            sizes = sorted(codes)
            
            getter = getattr(attributes.Affix, 'get_%s' % affix)
            for size in sizes:
                
                # size=size because if we don't use it, lambda sticks to the last value of 
                # the loop iterator size
                def f(word, size=size):
                    return getter(word, size)
                
                self.converter.add_extractor(f)
        
        self.converter = attributes.TokenConverter()
        self.converter.add_extractor(self.word_dict.get)
        if metadata.use_caps:
            self.converter.add_extractor(get_capitalization)
        if metadata.use_prefix:
            add_affix_extractors('prefix')
        if metadata.use_suffix:
            add_affix_extractors('suffix')
        

class TaggerReader(TextReader):
    """
    Abstract class extending TextReader with useful functions
    for tagging tasks. 
    """
    
    def __init__(self, load_dictionaries=True):
        '''
        This class shouldn't be used directly. The constructor only
        provides method calls for subclasses.
        '''
        if load_dictionaries:
            self.load_dictionary()
            self.load_tag_dict()
        
        self.codified = False
    
    def load_or_create_dictionary(self):
        """
        Try to load the vocabulary from the default location. If the vocabulary
        file is not available, create a new one from the sentences available
        and save it.
        """
        if os.path.isfile(config.FILES['vocabulary']):
            self.load_dictionary()
            return
        
        self.generate_dictionary(minimum_occurrences=2)
        self.save_dictionary()
    
    def load_or_create_tag_dict(self):
        """
        Try to load the tag dictionary from the default location. If the dictinaty
        file is not available, scan the available sentences and create a new one. 
        """
        key = '%s_tag_dict' % self.task
        filename = config.FILES[key]
        if os.path.isfile(filename):
            self.load_tag_dict(filename)
            return
        
        tags = {tag for sent in self.sentences for _, tag in sent}
        self.tag_dict = {tag: code for code, tag in enumerate(tags)}
        self.save_dictionary(filename)
    
    def generate_dictionary(self, dict_size=None, minimum_occurrences=2):
        """
        Generates a token dictionary based on the given sentences.
        
        :param dict_size: Max number of tokens to be included in the dictionary.
        :param minimum_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary. 
        """
        logger = logging.getLogger("Logger")
                
        tokens = [token for sent in self.sentences for token, _ in sent]
        self.word_dict = WordDictionary(tokens, dict_size, minimum_occurrences)
        logger.info("Created dictionary with %d types" % self.word_dict.num_tokens)
        
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
    
    def save_tag_dict(self, tag_dict=None, filename=None):
        """
        Saves a tag dictionary to a file as a list of tags.
        
        :param tag_dict: the dictionary to save. If None, the default
            tag_dict for the class will be saved.
        :param filename: the file where the dictionary should be saved.
            If None, the class default tag_dict filename will be used.
        """
        if tag_dict is None:
            tag_dict = self.tag_dict
        if filename is None:
            key = '%s_tag_dict' % self.task
            filename = config.FILES[key]
        
        save_tag_dict(tag_dict, filename)
    
    def load_tag_dict(self, filename=None):
        """
        Load the tag dictionary from the default file and assign
        it to the tag_dict attribute. 
        """
        if filename is None:
            key = '%s_tag_dict' % self.task
            filename = config.FILES[key]
            
        self.tag_dict = load_tag_dict(filename)
    
    
    