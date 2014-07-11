# -*- coding: utf-8 -*-

'''
Class for dealing with dependency parsing data.
'''

import os
import logging
import numpy as np

from .. import attributes
from .. import config
from .. import reader

class ConllPos(object):
    '''
    Dummy class to store field positions in a CoNLL-like file
    for dependency parsing. NB: The positions are different from
    those used in SRL!
    '''
    id = 0
    word = 1
    lemma = 2
    pos = 3
    pos2 = 4
    morph = 5
    dep_head = 6 # dependency head
    dep_rel = 7 # dependency relation


class DependencyReader(reader.TaggerReader):

    def __init__(self, md=None, filename=None):
        '''
        Constructor
        '''
        if filename is not None:
            self._read_conll(filename)
        
        self._set_metadata(md)
        self.task = 'dependency'
        self.load_dictionary()
        self.rare_tag = None
        self.pos_dict = None
        
    
    def _read_conll(self, filename): 
        '''
        Read data from a CoNLL formatted file.
        '''
        lines = []
        self.sentences = []
        self.heads = []
        
        # this keeps track of the tokens 
        sentence = []
        
        # this has the number of each token's head, in the same order as 
        # the tokens appear
        sentence_heads = []
        
        with open(filename, 'rb') as f:
            for line in f:
                line = unicode(line, 'utf-8').strip()
                lines.append(line)
        
        for line in lines:
            if line == '':
                # empty line, last sentence is finished 
                if len(sentence) > 0:
                    self.sentences.append(sentence)
                    self.heads.append(np.array(sentence_heads))
                    sentence = []
                    sentence_heads = []
                    
                continue
            
            fields = line.split()
            word = fields[ConllPos.word]
            pos = fields[ConllPos.pos]
            head = int(fields[ConllPos.dep_head])
            
            if head == 0:
                # we represent a dependency to root as an edge to the token itself
                head = int(fields[ConllPos.id]) 
            
            # -1 because tokens are numbered from 1
            head -= 1
            
            token = attributes.Token(word, pos=pos)
            sentence.append(token)
            sentence_heads.append(head)
    
    def _create_pos_dict(self):
        """
        Examine all POS tags in the sentences and create a dictionary based on them.
        """
        logger = logging.getLogger("Logger")
        logger.info('Creating new POS tag dictionary (for dependency parsing)')
        tags = {token.pos for sent in self.sentences
                for token in sent}
        pos_dict = {tag: code for code, tag in enumerate(tags)}
        
        return pos_dict
    
    def load_pos_dict(self):
        """
        Load the pos tag dictionary (specific for dependency parsing) 
        from its default location.
        """
        logger = logging.getLogger("Logger")
        logger.debug('Loading POS tag dictionary (for dependency parsing)')
        pos_dict = reader.load_tag_dict(config.FILES['pos_tags'])
        return pos_dict
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        Tags are left as the index of the each token's head.
        """
        new_sentences = []
        for sent in self.sentences:
            new_sentence = []
            
            for token in sent:
                new_token = self.converter.convert(token)
                new_sentence.append(new_token)
            
            new_sentences.append(np.array(new_sentence))
        
        self.sentences = new_sentences
        self.codified = True
    
    def load_or_create_pos_dict(self):
        """
        Try to load the pos tag dictionary to be used with this reader (when
        using POS tags as additional features). If there isn't a file in the 
        data directory with the right name, a new dictionary is created 
        after examining the data.
        """
        if self.pos_dict is not None:
            return
        
        if os.path.isfile(config.FILES['pos_tags']):
            self.pos_dict = self.load_pos_dict()
        else:
            self.pos_dict = self._create_pos_dict()
            self.save_tag_dict(self.pos_dict, config.FILES['pos_tags'])
        
        # adding the padding pos key must come after saving the dictionary
        # because that key shouldn't be used in a POS tagger that shares the 
        # pos dict
        if attributes.PADDING_POS not in self.pos_dict:
            code = max(self.pos_dict.values()) + 1
            self.pos_dict[attributes.PADDING_POS] = code
    
    def get_num_pos_tags(self):
        """
        Return the number of POS tags that can be used as an additional feature
        by this reader.
        """
        self._load_or_create_pos_dict()
        return len(self.pos_dict)
    
    def create_converter(self, metadata):
        """
        This function overrides the TextReader's one in order to deal with Token
        objects instead of raw strings. It also allows POS as an attribute.
        """
        f = lambda token: self.word_dict[token.word]
        self.converter = attributes.TokenConverter()
        self.converter.add_extractor(f)
        
        if metadata.use_caps:
            caps_lookup = lambda t: attributes.get_capitalization(t.word)
            self.converter.add_extractor(caps_lookup)
        
        if metadata.use_pos:
            self._load_or_create_pos_dict()
            g = lambda token: self.pos_dict[token.pos]
            self.converter.add_extractor(g)
            
        