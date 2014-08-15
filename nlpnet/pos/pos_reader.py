# -*- coding: utf-8 -*-

"""
Class for dealing with POS data.
"""

from .. import utils
from ..reader import TaggerReader

class POSReader(TaggerReader):
    """
    This class reads data from a POS corpus and turns it into a format
    readable by the neural network for the POS tagging task.
    """
    
    def __init__(self, md=None, sentences=None, filename=None):
        """
        :param tagged_text: a sequence of tagged sentences. Each sentence must be a 
            sequence of (token, tag) tuples. If None, the sentences are read from the 
            default location.
        """
        self.md = md
        self.task = 'pos'
        self.rare_tag = None
                
        self._set_metadata(md)
                
        if sentences is not None:
            self.sentences = sentences
        else:
            self.sentences = []
            
            if filename is not None:
                with open(filename, 'rb') as f:
                    for line in f:
                        line = unicode(line, 'utf-8')
                        items = line.split()
                        sentence = []
                        for item in items:
                            token, tag = item.rsplit('_', 1)
                            sentence.append((token, tag))
                            
                        self.sentences.append(sentence)
            

# backwards compatibility
MacMorphoReader = POSReader
