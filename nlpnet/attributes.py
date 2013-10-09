# -*- coding: utf-8 -*-

import logging
import numpy as np
import re
from collections import Counter

import config
from word_dictionary import WordDictionary as WD

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
    codes = {}
    # words smaller than the suffix size
    small_word = 0
    other = 1
    num_suffixes = 2
    
    @classmethod
    def load_suffixes(cls):
        """
        loads the listed suffixes from the suffix file.
        """
        Suffix.codes = {}
        code = Suffix.other + 1
        logger = logging.getLogger("Logger")
        try:
            with open(config.FILES['suffixes'], 'rb') as f:
                for line in f:
                    suffix = unicode(line.strip(), 'utf-8')
                    Suffix.codes[suffix] = code
                    code += 1
            Suffix.suffix_size = len(suffix)
            Suffix.num_suffixes = code 
        except IOError:
            logger.warning('Suffix list doesn\'t exist.')
            raise
    
    @classmethod
    def create_suffix_list(cls, wordlist, num, size, min_occurrences):
        """
        Creates a file containing the list of the most common suffixes found in 
        wordlist.
        
        :param wordlist: a list of word types (there shouldn't be repetitions)
        :param num: maximum number of suffixes
        :param size: desired size of suffixes
        :param min_occurrences: minimum number of occurrences of each suffix
        in wordlist
        """
        all_endings = [x[-size:] for x in wordlist 
                       if len(x) > size
                       and not re.search('_|\d', x[-size:])]
        c = Counter(all_endings)
        common_endings = c.most_common(num)
        suffix_list = [e for e, n in common_endings if n >= min_occurrences]
        
        with open(config.FILES['suffixes'], 'wb') as f:
            for suffix in suffix_list:
                f.write('%s\n' % suffix.encode('utf-8'))

    @classmethod
    def get_suffix(cls, word):
        """
        Returns the suffix code for the given word.
        """
        if len(word) < Suffix.suffix_size: return Suffix.small_word
        
        suffix = word[-Suffix.suffix_size:]
        return Suffix.codes.get(suffix.lower(), Suffix.other)

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

