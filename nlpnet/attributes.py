# -*- coding: utf-8 -*-

import logging
import numpy as np

import config
from word_dictionary import WordDictionary as WD
from collections import defaultdict

class Caps(object):
    """Dummy class for storing numeric values for capitalization."""
    num_values = 4
    lower = 0
    title = 1
    non_alpha = 2
    other = 3


class Token(object):
    def __init__(self, word, lemma='NA', pos='NA', morph='NA', chunk='NA'):
        """
        A token representation that stores discrete attributes to be given as 
        input to the neural network. 
        """
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.morph = morph
        self.chunk = chunk
    
    def __str__(self):
        return str(self.word)
    
    def __repr__(self):
        return self.word.__repr__()


class Suffix(object):
    """Dummy class for manipulating suffixes and their related codes."""
    # codes maps integers (suffix sizes) to dicts. each dict maps a suffix of the given 
    # size to its code
    codes = {}
    other = 0
    
    # initially, there is only the "other" (rare) suffix
    num_suffixes = 1
    
    @classmethod
    def load_suffixes(cls):
        """
        loads the listed suffixes from the suffix file.
        """
        Suffix.codes = {}
        logger = logging.getLogger("Logger")
        
        # intermediate storage
        suffixes_by_size = defaultdict(list)
        
        try:
            with open(config.FILES['suffixes'], 'rb') as f:
                for line in f:
                    suffix = unicode(line.strip(), 'utf-8')
                    size = len(suffix)
                    suffixes_by_size[size].append(suffix)
        except IOError:
            logger.error('Suffix list doesn\'t exist.')
            raise
        
        for size in suffixes_by_size:
            # for each size, each suffix has a code starting from 1
            # 0 is reserved for unknown suffixes
            Suffix.codes[size] = {suffix: code 
                                  for code, suffix in enumerate(suffixes_by_size[size], 1)}
        
        Suffix.num_sizes = len(suffixes_by_size)
        Suffix.num_suffixes_per_size = {size: len(Suffix.codes[size]) 
                                        for size in Suffix.codes}    
    
    @classmethod
    def get_suffix(cls, word):
        """
        Returns the suffix code for the given word.
        
        This implementation is not the most efficient (it checks every 
        suffix separately, instead of using a tree structure), but 
        it is intended to be called only once per word.
        """
        # check longer suffixes first
        for suffix in Suffix.suffixes_by_size:
            if word.endswith(suffix) and len(word) > len(suffix):
                return Suffix.codes[suffix]
        
        return Suffix.other


class TokenConverter(object):
    
    def __init__(self):
        """
        Class to convert tokens into indices to their feature vectos in
        feature matrices.
        """
        self.extractors = []
    
    def add_extractor(self, extractor):
        """
        Adds an extractor function to the TokenConverter. In order to get a token's 
        feature indices, the Converter will call each of its extraction functions passing
        the token as a parameter. The result will be a list containing each result. 
        """
        self.extractors.append(extractor)
    
    def get_padding_left(self, tokens_as_string=True):
        """
        Returns an object to be used as the left padding in the sentence.
        
        :param tokens_as_string: if True, treat tokens as strings. 
        If False, treat them as Token objects.
        """
        if tokens_as_string:
            pad = WD.padding_left
        else:
            pad = Token(WD.padding_left)
        return self.convert(pad)
    
    def get_padding_right(self, tokens_as_string=True):
        """
        Returns an object to be used as the right padding in the sentence.
        
        :param tokens_as_string: if True, treat tokens as strings. 
            If False, treat them as Token objects.
        """
        if tokens_as_string:
            pad = WD.padding_right
        else:
            pad = Token(WD.padding_right)
        return self.convert(pad)
    
    def convert(self, token):
        """
        Converts a token into its feature indices.
        """
        indices = np.array([function(token) for function in self.extractors])
        return indices

# capitalization
def get_capitalization(word):
    """
    Returns a code describing the capitalization of the word:
    lower, title, other or non-alpha (numbers and other tokens that can't be
    capitalized).
    """
    if not word.isalpha():
        return Caps.non_alpha
    
    if word.istitle():
        return Caps.title
    
    if word.islower():
        return Caps.lower
    
    return Caps.other

def capitalize(word, capitalization):
    """
    Capitalizes the word in the desired format. If the capitalization is 
    Caps.other, it is set all uppercase.
    """
    if capitalization == Caps.non_alpha:
        return word
    elif capitalization == Caps.lower:
        return word.lower()
    elif capitalization == Caps.title:
        return word.title()
    elif capitalization == Caps.other:
        return word.upper()
    else:
        raise ValueError("Unknown capitalization type.")

