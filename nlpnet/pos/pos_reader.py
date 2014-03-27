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
    
    def __init__(self, sentences=None, filename=None, load_dictionaries=True):
        """
        :param tagged_text: a sequence of tagged sentences. Each sentence must be a 
            sequence of (token, tag) tuples. If None, the sentences are read from the 
            default location.
        """
        self.task = 'pos'
        self.rare_tag = None
        super(POSReader, self).__init__(load_dictionaries)
        
        if sentences is not None:
            self.sentences = sentences
        else:
            self.sentences = []
            
            if filename is not None:
                with open(filename, 'rb') as f:
                    for line in f:
                        cleaned = utils.clean_text(unicode(line, 'utf-8'), False)
                        items = cleaned.split()
                        self.sentences.append([item.rsplit('_', 1) for item in items])
            

    def get_inverse_tag_dictionary(self):
        """
        Returns a version of the tag dictionary useful for consulting
        the meaning of the network's output.
        """
        tuples = [(x[1], x[0]) for x in self.tag_dict.iteritems()]
        ret = dict(tuples)
        
        return ret

# backwards compatibility
MacMorphoReader = POSReader
