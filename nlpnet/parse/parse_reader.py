# -*- coding: utf-8 -*-

'''
Class for dealing with dependency parsing data.
'''

import os
import logging
import numpy as np

from nlpnet import attributes
from nlpnet import reader
from nlpnet.word_dictionary import WordDictionary


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
    '''
    Class to read dependency files in CoNLL X format.
    '''
    
    def __init__(self, md=None, filename=None, labeled=False):
        '''
        Constructor.
        :param md: Metadata object containing the description for this reader
        :param filename: file containing data to be read and used in training
            or tagging
        :param labeled: (ignored if md is supplied) whether it is intended 
            to be used in labeled dependency parsing. Note that if it is 
            True, another reader object will be needed for unlabeled dependency.
        '''
        if md is not None:
            self.labeled = md.task.startswith('labeled')
        else:
            self.labeled = labeled
        
        if filename is not None:
            self._read_conll(filename)
        
        if self.labeled:
            self.taskname = 'labeled_dependency'
        else:
            self.taskname = 'unlabeled_dependency'
        
        self.rare_tag = None
        self.pos_dict = None
        super(DependencyReader, self).__init__(md)
        
    @property
    def task(self):
        """
        Abstract Base Class (ABC) attribute.
        """
        return self.taskname
                
    
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
        if self.labeled:
            self.labels = []
            sentence_labels = []
        
        with open(filename, 'rb') as f:
            for line in f:
                line = line.decode('utf-8').strip()
                lines.append(line)
        
        for line in lines:
            if line == '':
                # empty line, last sentence is finished 
                if len(sentence) > 0:
                    self.sentences.append(sentence)
                    self.heads.append(np.array(sentence_heads))
                    
                    if self.labeled:
                        self.labels.append(sentence_labels)
                        sentence_labels = []
                        
                    sentence = []
                    sentence_heads = []
                    
                continue
            
            fields = line.split()
            word = fields[ConllPos.word]
            pos = fields[ConllPos.pos]
            pos2 = fields[ConllPos.pos2]
            head = int(fields[ConllPos.dep_head])
            label = fields[ConllPos.dep_rel]
            
            if head == 0:
                # we represent a dependency to root as an edge to the token itself
                head = int(fields[ConllPos.id])
            
            # -1 because tokens are numbered from 1
            head -= 1
            
            token = attributes.Token(word, pos=pos, pos2=pos2)
            sentence.append(token)
            sentence_heads.append(head)
            if self.labeled:
                sentence_labels.append(label)
        
        # in case there was not an empty line after the last sentence 
        if len(sentence) > 0:
            self.sentences.append(sentence)
            self.heads.append(np.array(sentence_heads))
            if self.labeled:
                self.labels.append(sentence_labels)
    
    def _create_pos_dict(self):
        """
        Examine all POS tags in the sentences and create a dictionary based on them.
        """
        logger = logging.getLogger("Logger")
        logger.info('Creating new POS tag dictionary (for dependency parsing)')
        tags = {token.pos for sent in self.sentences
                for token in sent}
        pos_dict = {tag: code for code, tag in enumerate(tags)}
        
        code = max(pos_dict.values()) + 1
        pos_dict[attributes.PADDING_POS] = code
        
        return pos_dict
    
    def load_pos_dict(self):
        """
        Load the pos tag dictionary (specific for dependency parsing) 
        from its default location.
        """
        logger = logging.getLogger("Logger")
        logger.debug('Loading POS tag dictionary (for dependency parsing)')
        pos_dict = reader.load_tag_dict(self.md.paths['dependency_pos_tags'])
        return pos_dict
    
    def load_tag_dict(self, filename=None):
        """
        Verify if this reader is for the unlabeled dependency task. If so, 
        it doesn't use a tag dictionary and the call is ignored.
        """
        if not self.labeled:
            return
        
        super(DependencyReader, self).load_tag_dict(filename)    
    
    def load_or_create_tag_dict(self):
        """
        Try to load the tag dictionary from the default location. If the dictinaty
        file is not available, scan the available sentences and create a new one.
        
        It only is needed in labeled dependency parsing. 
        """
        if not self.labeled:
            return
        
        logger = logging.getLogger('Logger')
        filename = self.md.paths['dependency_tag_dict']
        if os.path.isfile(filename):
            self.load_tag_dict(filename)
            logger.debug('Loaded dependency tag dictionary')
            return
        
        tags = {tag for sent_labels in self.labels for tag in sent_labels}
        self.tag_dict = {tag: code for code, tag in enumerate(tags)}
        
        reader.save_tag_dict(filename, self.tag_dict)
        logger.debug('Saved dependency tag dictionary')
        
    
    def generate_dictionary(self, dict_size=None, minimum_occurrences=2):
        """
        Generates a token dictionary based on the given sentences.
        
        :param dict_size: Max number of tokens to be included in the dictionary.
        :param minimum_occurrences: Minimum number of times that a token must
            appear in the text in order to be included in the dictionary.
        """
        logger = logging.getLogger("Logger")
        all_tokens = [token.word 
                      for sent in self.sentences
                      for token in sent]
        self.word_dict = WordDictionary(all_tokens, dict_size, minimum_occurrences)
        logger.info("Created dictionary with %d tokens" % self.word_dict.num_tokens)
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        Tags are left as the index of the each token's head.
        """
        if self.converter is None:
            self.create_converter()
        
        self.sentences = [np.array([self.converter.convert(token) for token in sent])
                          for sent in self.sentences]
        
        if self.labeled:
            self.labels = [np.array([self.tag_dict[label] for label in sent_labels]) 
                           for sent_labels in self.labels]
        
        self.codified = True
    
    def _load_or_create_pos_dict(self):
        """
        Try to load the pos tag dictionary to be used with this reader (when
        using POS tags as additional features). If there isn't a file in the 
        data directory with the right name, a new dictionary is created 
        after examining the data.
        """
        if self.pos_dict is not None:
            return
        
        if os.path.isfile(self.md.paths['dependency_pos_tags']):
            self.pos_dict = self.load_pos_dict()
        else:
            self.pos_dict = self._create_pos_dict()
            self.save_tag_dict(self.md.paths['dependency_pos_tags'], self.pos_dict)
        
    def get_num_pos_tags(self):
        """
        Return the number of POS tags that can be used as an additional feature
        by this reader.
        """
        self._load_or_create_pos_dict()
        return len(self.pos_dict)
    
    def create_converter(self):
        """
        This function overrides the TextReader's one in order to deal with Token
        objects instead of raw strings. It also allows POS as an attribute.
        """
        f = lambda token: self.word_dict[token.word]
        self.converter = attributes.TokenConverter()
        self.converter.add_extractor(f)
        
        if self.md.use_caps:
            caps_lookup = lambda t: attributes.get_capitalization(t.word)
            self.converter.add_extractor(caps_lookup)
        
        if self.md.use_pos:
            self._load_or_create_pos_dict()
            g = lambda token: self.pos_dict[token.pos]
            self.converter.add_extractor(g)
            
        