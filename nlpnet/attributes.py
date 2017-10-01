# -*- coding: utf-8 -*-

import logging
import numpy as np

from nlpnet.word_dictionary import WordDictionary as WD
from collections import defaultdict

# dummy value to be used when POS is an additional attribute
PADDING_POS = 'PADDING'


class Caps(object):
    """Dummy class for storing numeric values for capitalization."""
    num_values = 5
    lower = 0
    title = 1
    non_alpha = 2
    other = 3
    padding = 4


class Token(object):
    def __init__(self, word, lemma='NA', pos='NA', pos2='NA', morph='NA',
                 chunk='NA'):
        """
        A token representation that stores discrete attributes to be given as 
        input to the neural network. 
        """
        self.word = word
        self.lemma = lemma
        self.pos = pos
        self.pos2 = pos2
        self.morph = morph
        self.chunk = chunk
    
    def __str__(self):
        return str(self.word)
    
    def __repr__(self):
        return self.word.__repr__()


class Affix(object):
    """Dummy class for manipulating suffixes and their related codes."""
    # codes maps integers (affix sizes) to dicts.
    # each dict maps a suffix of the given size to its code
    suffix_codes = {}
    prefix_codes = {}
    other = 0
    padding = 1
    num_suffixes_per_size = {}
    num_prefixes_per_size = {}
    
    @classmethod
    def load_suffixes(cls, md):
        """
        Loads suffixes from the suffix file.
        """
        cls.load_affixes(cls.suffix_codes, md.paths['suffixes'])
        
        # +2 because of the unkown suffix code and padding
        cls.num_suffixes_per_size = {size: len(cls.suffix_codes[size]) + 2
                                     for size in cls.suffix_codes}
    
    @classmethod
    def load_prefixes(cls, md):
        """
        Loads prefixes from the prefix file.
        """
        cls.load_affixes(cls.prefix_codes, md.paths['prefixes'])
        
        # +2 because of the unkown prefix code and padding
        cls.num_prefixes_per_size = {size: len(cls.prefix_codes[size]) + 2
                                     for size in cls.prefix_codes}

    @classmethod
    def load_affixes(cls, codes, filename):
        """
        Parent function for loading prefixes and suffixes.
        """
        logger = logging.getLogger("Logger")
        
        # intermediate storage
        affixes_by_size = defaultdict(list)
        
        try:
            with open(filename, 'rb') as f:
                for line in f:
                    affix = line.strip().decode('utf-8')
                    size = len(affix)
                    affixes_by_size[size].append(affix)
        except IOError:
            logger.error("File %s doesn't exist." % filename)
            raise
        
        for size in affixes_by_size:
            # for each size, each affix has a code starting from 2
            # 0 is reserved for unknown affixes
            # 1 is reserved for padding pseudo-affixes
            codes[size] = {affix: code 
                           for code, affix in enumerate(affixes_by_size[size],
                                                        2)}
    
    @classmethod
    def get_suffix(cls, word, size):
        """
        Return the suffix code for the given word. Consider a suffix
        of the given size.
        """
        if word == WD.padding_left or word == WD.padding_right:
            return cls.padding
        
        if len(word) <= size:
            return cls.other
        
        suffix = word[-size:].lower()
        code = cls.suffix_codes[size].get(suffix, cls.other)
        return code
    
    @classmethod
    def get_prefix(cls, word, size):
        """
        Return the suffix code for the given word. Consider a suffix
        of the given size.
        """
        if word == WD.padding_left or word == WD.padding_right:
            return cls.padding
        
        if len(word) <= size:
            return cls.other
        
        prefix = word[:size].lower()
        code = cls.prefix_codes[size].get(prefix, cls.other)
        return code


class TokenConverter(object):
    
    def __init__(self):
        """
        Class to convert tokens into indices to their feature vectos in
        feature matrices.
        """
        self.extractors = []
    
    def add_extractor(self, extractor):
        """
        Adds an extractor function to the TokenConverter.
        In order to get a token's feature indices, the Converter will call each
        of its extraction functions passing the token as a parameter.
        The result will be a list containing each result.
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
            pad = Token(WD.padding_left, pos=PADDING_POS)
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
            pad = Token(WD.padding_right, pos=PADDING_POS)
        return self.convert(pad)
    
    def convert(self, token):
        """
        Converts a token into its feature indices.
        """
        indices = np.array([function(token) for function in self.extractors])
        return indices


def get_capitalization(word):
    """
    Returns a code describing the capitalization of the word:
    lower, title, other or non-alpha (numbers and other tokens that can't be
    capitalized).
    """
    if word == WD.padding_left or word == WD.padding_right:
        return Caps.padding
    
    if not any(c.isalpha() for c in word):
        # check if there is at least one letter
        # (this is faster than using a regex)
        return Caps.non_alpha
    
    if word.islower():
        return Caps.lower
    
    # word.istitle() returns false for compunds like Low-cost
    if len(word) == 1:
        # if we reached here, there's a single upper case letter
        return Caps.title
    elif word[0].isupper() and word[1:].islower():
        return Caps.title    
    
    return Caps.other


def capitalize(word, capitalization):
    """
    Capitalizes the word in the desired format. If the capitalization is 
    Caps.other, it is set all uppercase.
    """
    if capitalization == Caps.non_alpha or capitalization == Caps.padding:
        return word
    elif capitalization == Caps.lower:
        return word.lower()
    elif capitalization == Caps.title:
        return word[0].upper() + word[1:].lower()
    elif capitalization == Caps.other:
        return word.upper()
    else:
        raise ValueError("Unknown capitalization type.")

