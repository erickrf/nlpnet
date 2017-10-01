#!/usr/env python
# -*- coding: utf-8 -*-

"""
Base class for reading NLP tagging data.
"""

import os
import re
import abc
import logging
import numpy as np
from collections import Counter

from nlpnet import attributes
from nlpnet import metadata
from nlpnet import config
from nlpnet.word_dictionary import WordDictionary
from nlpnet.attributes import get_capitalization


class FileNotFoundException(IOError):
    """
    Dummy class for indicating file not found instead of 
    the broad IOError.
    """
    pass


def load_tag_dict(filename):
    """
    Load a tag dictionary from a file containing one tag
    per line.
    """
    tag_dict = {}
    with open(filename, 'rb') as f:
        code = 0
        for tag in f:
            tag = tag.decode('utf-8').strip()
            if tag:
                tag_dict[tag] = code
                code += 1
    
    return tag_dict


def save_tag_dict(filename, tag_dict):
    """
    Save the given tag dictionary to the given file. Dictionary
    is saved with one tag per line, in the order of their codes.
    """
    ordered_keys = sorted(tag_dict, key=tag_dict.get)
    text = '\n'.join(ordered_keys)
    with open(filename, 'wb') as f:
        f.write(text.encode('utf-8'))


class TaggerReader(object):
    """
    Abstract class extending TextReader with useful functions
    for tagging tasks. 
    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, md=None, load_dictionaries=True):
        '''
        This class shouldn't be used directly. The constructor only
        provides method calls for subclasses. Subclasses should call
        this constructor after initializing the `task` attribute.
        '''
        self._set_metadata(md)
        self.codified = False
        self._converter = None
        
        if load_dictionaries:
            self.load_or_create_dictionary()
            self.load_or_create_tag_dict()
    
    @abc.abstractmethod
    def task(self):
        """
        The task the tagger reads data for.
        Must be defined in subclasses.
        """
        return None
    
    def load_or_create_dictionary(self):
        """
        Try to load the vocabulary from the default location. If the vocabulary
        file is not available, create a new one from the sentences available
        and save it.
        """
        try:
            self.load_dictionary()
        except FileNotFoundException:
            self.generate_dictionary(minimum_occurrences=2)
            self.save_dictionary()
    
    def load_or_create_tag_dict(self):
        """
        Try to load the tag dictionary from the default location. If the dictinaty
        file is not available, scan the available sentences and create a new one. 
        """
        key = '%s_tag_dict' % self.task
        filename = self.md.paths[key]
        if os.path.isfile(filename):
            self.load_tag_dict(filename)
            return
        
        tags = {tag for sent in self.sentences for _, tag in sent}
        self.tag_dict = {tag: code for code, tag in enumerate(tags)}
        self.save_tag_dict(filename)
    
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
        logger.info("Created dictionary with %d types" %
                    self.word_dict.num_tokens)
        
    def get_inverse_tag_dictionary(self):
        """
        Returns a version of the tag dictionary that maps numbers to tags.
        Used for consulting the meaning of the network's output.
        """
        tuples = [(x[1], x[0]) for x in self.tag_dict.items()]
        ret = dict(tuples)
        
        return ret
    
    def codify_sentence(self, sentence):
        """
        Converts a given sentence into the indices used by the neural network.
        
        :param sentence: a sequence of tokens, already tokenized
        """
        if self._converter is None:
            self.create_converter()
        return np.array([self.converter.convert(t) for t in sentence])
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        """
        if self._converter is None:
            self.create_converter()
        
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
        c = Counter(token.lower()
                    for sent in self.sentences for token, _ in sent)
        return c
    
    def get_tag_counter(self):
        """
        Returns a Counter object with tag occurrences.
        """
        c = Counter(tag for sent in self.sentences for _, tag in sent)
        return c
    
    def save_tag_dict(self, filename=None, tag_dict=None):
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
            filename = self.md.paths[key]
        
        save_tag_dict(filename, tag_dict)
    
    def load_tag_dict(self, filename=None):
        """
        Load the tag dictionary from the default file and assign
        it to the tag_dict attribute. 
        """
        if filename is None:
            key = '%s_tag_dict' % self.task
            filename = self.md.paths[key]
            
        self.tag_dict = load_tag_dict(filename)
       
    def _set_metadata(self, md):
        if md is None:
            #metadata not provided = using global data_dir for files
            self.md = metadata.Metadata(self.task, config.FILES)
        else:
            self.md = md
        
    def add_text(self, text):
        """
        Adds more text to the reader. The text must be a sequence of sequences
        of tokens.
        """
        self.sentences.extend(text)
    
    def load_dictionary(self):
        """Read a file with a word list and create a dictionary."""
        logger = logging.getLogger("Logger")
        logger.info("Loading vocabulary")
        
        # try to load vocabulary specific for the task
        key = 'vocabulary_%s' % self.task
        filename = self.md.paths[key]
        if not os.path.isfile(filename):
            # fallback to generic vocabulary
            filename = self.md.paths['vocabulary']
            if not os.path.isfile(filename):
                raise FileNotFoundException()
        
        words = []
        with open(filename, 'rb') as f:
            for word in f:
                word = word.decode('utf-8').strip()
                if word:
                    words.append(word)
        
        wd = WordDictionary.init_from_wordlist(words)
        self.word_dict = wd
        logger.debug("Done. Dictionary size is %d types" % wd.num_tokens)
    
    def save_dictionary(self, filename=None):
        """
        Saves the reader's word dictionary as a list of words.
        
        :param filename: path to the file to save the dictionary. 
            if not given, it will be saved in the default nlpnet
            data directory.
        """
        logger = logging.getLogger("Logger")
        if filename is None:
            key = 'vocabulary_%s' % self.task
            filename = self.md.paths[key]
        
        self.word_dict.save(filename)
        logger.info("Dictionary saved in %s" % filename)
    
    def create_affix_list(self, prefix_or_suffix, max_size, min_occurrences):
        """
        Handle the creation of suffix and prefix lists.
        
        Check if there exists an affix list in the data directory. If there
        isn't, create a new one based on the training sentences.
        
        :param prefix_or_suffix: string 'prefix' or 'suffix'
        """
        affix_type = prefix_or_suffix.lower()
        assert affix_type == 'suffix' or affix_type == 'prefix' 
        
        filename = self.md.paths['%ses' % affix_type]
        if os.path.isfile(filename):
            return
        
        logger = logging.getLogger("Logger")
        affixes_all_lengths = []
        
        # only get the affix size n from words with length at least (n+1)
        types = {re.sub(r'\d', '9', token.lower()) 
                 for sent in self.sentences for token, _ in sent}
        
        for length in range(1, max_size + 1):
            if affix_type == 'suffix':
                c = Counter(type_[-length:]
                            for type_ in types
                            if len(type_) > length)
            else:
                c = Counter(type_[:length]
                            for type_ in types
                            if len(type_) > length)
            affixes_this_length = [affix for affix in c 
                                   if c[affix] >= min_occurrences]
            affixes_all_lengths.extend(affixes_this_length)
        
        logger.info('Created a list of %d %ses.' % (len(affixes_all_lengths),
                                                    affix_type))
        text = '\n'.join(affixes_all_lengths)
        with open(filename, 'wb') as f:
            f.write(text.encode('utf-8'))
    
    @property
    def converter(self):
        """
        Return the token converter, which transforms tokens into their feature
        vector indices. If it doesn't exist, one is created. 
        """
        if self._converter is None:
            self.create_converter()
        
        return self._converter
    
    @converter.setter
    def converter(self, value):
        self._converter = value
    
    def create_converter(self):
        """
        Sets up the token converter, which is responsible for transforming
        tokens into their feature vector indices
        """
        def add_affix_extractors(affix):
            """
            Helper function that works for both suffixes and prefixes.
            The parameter affix should be 'suffix' or 'prefix'.
            """
            loader_function = getattr(attributes.Affix, 'load_%ses' % affix)
            loader_function(self.md)
            
            # deal with gaps between sizes (i.e., if there are sizes 2, 3,
            # and 5)
            codes = getattr(attributes.Affix, '%s_codes' % affix)
            sizes = sorted(codes)
            
            getter = getattr(attributes.Affix, 'get_%s' % affix)
            for size in sizes:
                
                # size=size because if we don't use it, lambda sticks to the
                # last value of the loop iterator size
                def f(word, size=size):
                    return getter(re.sub(r'\d', '9', word), size)
                
                self.converter.add_extractor(f)
        
        self._converter = attributes.TokenConverter()
        self.converter.add_extractor(self.word_dict.get)
        if self.md.use_caps:
            self.converter.add_extractor(get_capitalization)
        if self.md.use_prefix:
            add_affix_extractors('prefix')
        if self.md.use_suffix:
            add_affix_extractors('suffix')
