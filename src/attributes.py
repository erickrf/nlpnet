# -*- coding: utf-8 -*-

import logging
import numpy as np
from collections import Counter

import config
from word_dictionary import WordDictionary as WD

class Caps(object):
    """
    Dummy class for storing numeric values for capitalization.
    """
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
    """
    Dummy class for manipulating suffixes and their related codes.
    """
    codes = {}
    @classmethod
    def init(cls):
        # loads the listed suffixes
        code = 1
        logger = logging.getLogger("Logger")
        try:
            with open(config.FILES['suffixes']) as f:
                for line in f:
                    suffix = unicode(line.strip(), 'utf-8')
                    Suffix.codes[suffix] = code
                    code += 1
        except IOError:
            logger.warning('Suffix list doesn\'t exist.')


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
        @param tokens_as_string: if True, treat tokens as strings. 
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
        @param tokens_as_string: if True, treat tokens as strings. 
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


def get_suffix(word):
    """
    Returns a number corresponding to the last few letters of a word.
    """
    if len(word) > 3:
        ending = word[-3:]
        return Suffix.codes.get(ending, 0)
    return 0
    

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

def build_suffix_list(wordlist, num=40, max_size=3):
    suffix_list = common_suffixes(wordlist, max_size, num)
    with open(config.FILES['suffixes'], 'wb') as f:
        for suffix in suffix_list:
            f.write('%s\n' % suffix.encode('utf-8'))

def common_suffixes(wordlist, size, num):
    """
    Returns the most common suffixes with a given size
    in the wordlist.
    """
    logger = logging.getLogger('Logger')
    endings = [x[-size:] for x in wordlist 
               if len(x) > size
               and '_' not in x[-size:]]
    c = Counter(endings)
    top = c.most_common(num)
    suffixes = [x[0] for x in top]
    occurrences = [x[1] for x in top]

    logger.info('%d occurrences for the most common suffix. %d for the least common' % (occurrences[0], occurrences[-1]))
    logger.info('%d occurences per suffix in average' % float(sum(occurrences)/len(occurrences)))
    
    return suffixes

# initializes suffix data
#Suffix.init()


