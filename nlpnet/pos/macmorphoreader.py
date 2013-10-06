# -*- coding: utf-8 -*-

"""
Class for dealing with POS data from MacMorpho.
"""

import cPickle

import config
from reader import TaggerReader

class MacMorphoReader(TaggerReader):
    """
    This class reads data from the MacMorpho corpus and turns it into a format
    readable by the neural network for the POS tagging task.
    """
    
    def __init__(self, sentences=None, filename=None):
        """
        @param tagged_text: a sequence of tagged sentences. Each sentence must be a 
        sequence of (token, tag) tuples. If None, the sentences are read from the 
        default location.
        """
        self.task = 'pos'
        self.rare_tag = None
        
        if sentences is not None:
            self.sentences = sentences
        else:
            self.sentences = []
            
            if filename is not None:
                with open(filename, 'rb') as f:
                    for line in f:
                        items = unicode(line, 'utf-8').split()
                        self.sentences.append([item.split('_') for item in items])
                        
            
            """
            To read sentences from scratch, use the following: 
            
            self.sentences = []
            
            for root, _, files in os.walk(config.DIRS['macmorpho']):
                for file_ in files:
                    filename = os.path.join(root, file_)
                    file_sents = pos.train_pos.read_macmorpho_file(filename)
                    self.sentences.extend(file_sents)
            
            Then set aside every 10th sentence to test
            """     

    def get_inverse_tag_dictionary(self):
        """
        Returns a version of the tag dictionary useful for consulting
        the meaning of the network's output.
        """
        tuples = [(x[1], x[0]) for x in self.tag_dict.iteritems()]
        ret = dict(tuples)
        
        return ret
    
