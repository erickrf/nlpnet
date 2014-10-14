# -*- coding: utf-8 -*-

"""
A neural network for NLP tagging tasks such as dependency
parsing, where each token has another (or root) as a head. 
"""


import numpy as np
cimport numpy as np

cdef class DependencyNetwork2(Network):
    
    # weights connecting each part of the input to the hidden layers
    cdef readonly np.ndarray head_weights, modifier_weights, distance_weights
    
    # the scores of all possible dependency edges among tokens
    cdef readonly np.ndarray dependency_scores
    
    def tag_sentence(self, np.ndarray sentence, np.ndarray heads=None):
        """
        Run the network for the unlabeled dependency task.
        A graph with all weights for possible dependencies is built 
        and the final answer is obtained applying the Chu-Liu-Edmond's
        algorithm.
        """
        training = heads is not None
        
        self._create_sentence_lookup(sentence)
        self._create_hidden_lookup(sentence)
        
        num_tokens = len(sentence)
        # dependency_weights [i, j] has the score for token i having j as a head.
        # the main diagonal has the values for dependencies from the root and is 
        # later copied to the last column for easier processing
        self.dependency_weights = np.empty((num_tokens, num_tokens + 1))
        
        # calculate the score for each (head, modifier) pair
        for modifier in range(num_tokens):
            
            # pre load the hidden layer with values from the modifier
            input_data = self.sentence_lookup[i:i + self.word_window_size].flatten()
