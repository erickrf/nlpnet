# -*- coding: utf-8 -*-

"""
A neural network for NLP tagging tasks.
It employs feature tables to store feature vectors for each token.
"""

import numpy as np
cimport numpy as np
from cpython cimport bool
import math
from itertools import izip

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t
ctypedef Py_ssize_t INDEX_t

cdef class Network:
    
    # sizes and learning rates
    cdef readonly int word_window_size, input_size, hidden_size, output_size
    cdef public float learning_rate, learning_rate_features
    
    # padding stuff
    cdef np.ndarray padding_left, padding_right
    cdef np.ndarray pre_padding, pos_padding
    
    # weights, biases, calculated values
    cdef readonly np.ndarray hidden_weights, output_weights
    cdef readonly np.ndarray hidden_bias, output_bias
    cdef readonly np.ndarray input_values, hidden_values
    
    # feature_tables 
    cdef public list feature_tables
    
    # data for statistics during training. 
    cdef float error, accuracy
    cdef int total_items, train_hits, skips
    
    @classmethod
    def create_new(cls, feature_tables, int word_window, int hidden_size, 
                 int output_size):
        """
        Creates a new neural network.
        """
        # sum the number of features in all tables 
        cdef int input_size = sum(table.shape[1] for table in feature_tables)
        input_size *= word_window
        
        # creates the weight matrices
        # all weights are between -0.1 and +0.1
        hidden_weights = 0.2 * np.random.random((hidden_size, input_size)) - 0.1
        hidden_bias = 0.2 * np.random.random(hidden_size) - 0.1
        output_weights = 0.2 * np.random.random((output_size, hidden_size)) - 0.1
        output_bias = 0.2 * np.random.random(output_size) - 0.1
        
        net = Network(word_window, input_size, hidden_size, output_size,
                      hidden_weights, hidden_bias, output_weights, output_bias)
        net.feature_tables = feature_tables
        
        return net
        
    def __init__(self, word_window, input_size, hidden_size, output_size,
                 hidden_weights, hidden_bias, output_weights, output_bias):
        """
        This function isn't expected to be directly called.
        Instead, use the classmethods load_from_file or 
        create_new.
        """
        self.learning_rate = 0
        self.learning_rate_features = 0
        
        self.word_window_size = word_window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias
    
    def run(self, np.ndarray indices):
        """
        Runs the network for a given input. 
        @param indices: a 2-dim np array of indices to the feature tables.
        Each element must have the indices to each feature table.
        """
        # find the actual input values concatenating the feature vectors
        # for each input token
        cdef np.ndarray input_data
        input_data = np.concatenate(
                                    [table[index] 
                                     for token_indices in indices
                                     for index, table in zip(token_indices, 
                                                             self.feature_tables)
                                     ]
                                    )
        
        # store the output in self in order to use in the backprop
        self.input_values = input_data
        self.hidden_values = self.hidden_weights.dot(input_data)
        self.hidden_values += self.hidden_bias
        self.hidden_values = np.tanh(self.hidden_values)
        
        cdef np.ndarray output = self.output_weights.dot(self.hidden_values)
        output += self.output_bias
        
        return output
    
    property padding_left:
        """
        The padding element filling the "void" before the beginning
        of the sentence.
        """
        def __get__(self):
            return self.padding_left
    
        def __set__(self, np.ndarray padding_left):
            self.padding_left = padding_left
            self.pre_padding = np.array((self.word_window_size / 2) * [padding_left])
    
    property padding_right:
        """
        The padding element filling the "void" after the end
        of the sentence.
        """
        def __get__(self):
            return self.padding_right
    
        def __set__(self, np.ndarray padding_right):
            self.padding_right = padding_right
            self.pos_padding = np.array((self.word_window_size / 2) * [padding_right])
    
    def tag_sentence(self, np.ndarray sentence):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        @param sentence: a 2-dim numpy array, where each item
        encodes a token.
        """
        return self._tag_sentence(sentence, train=False)
    
    def _tag_sentence(self, np.ndarray sentence, bool train=False, tags=None):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        @param sentence: a 2-dim numpy array, where each item
        encodes a token.
        @param train: if True, perform weight and feature correction.
        @param tags: the correct tags (needed when training)
        """
        cdef list answer = []
        
        # add padding to the sentence
        cdef np.ndarray padded_sentence = np.vstack((self.pre_padding,
                                                     sentence,
                                                     self.pos_padding))

        # get the first window
        cdef np.ndarray window = padded_sentence[:self.word_window_size]
        
        cdef np.ndarray result = self.run(window)
        answer.append(result.argmax())
        
        cdef object iter_tags
        if train:
            iter_tags = iter(tags)
            self._corrections(window, result, iter_tags.next())
        
        # run for the rest of the windows in the sentence
        cdef np.ndarray element
        for element in padded_sentence[self.word_window_size:]:
            window = np.vstack((window[1:], element))
            result = self.run(window)
            answer.append(result.argmax())
            if train:
                self._corrections(window, result, iter_tags.next()) 
              
        return answer
    
    def train(self, list sentences, list tags, 
              int epochs, int epochs_between_reports=0,
              float desired_accuracy=0):
        """
        Trains the network to tag sentences.
        @param sentences: a list of 2-dim numpy arrays, where each item
        encodes a sentence. Each item in a sentence has the 
        indices to its features.
        @param tags: a list of 1-dim numpy arrays, where each item has
        the tags of the sentences.
        @param epochs: number of training epochs
        @param epochs_between_reports: number of epochs to wait between
        reports about the training performance. 0 means no reports.
        @param desired_accuracy: training stops if the desired accuracy
        is reached. Ignored if 0.
        """
        print "Training for up to %d epochs" % epochs
        
        for i in range(epochs):
            self._train_epoch(sentences, tags)
            
            self.accuracy = float(self.train_hits) / self.total_items
            
            if (epochs_between_reports > 0 and i % epochs_between_reports == 0) \
                or self.accuracy >= desired_accuracy > 0:
                
                self._print_epoch_report(i + 1)
                
                if self.accuracy >= desired_accuracy > 0:
                    break
            
        
        self.error = 0
        self.train_hits = 0
        self.total_items = 0
            
    def _print_epoch_report(self, int num):
        """
        Reports the status of the network in the given training
        epoch, including error and accuracy.
        """
        cdef float error = self.error / self.total_items
        print "%d epochs   Error:   %f   Accuracy: %f   %d corrections could be skipped" % (num,
                                                                                            error,
                                                                                            self.accuracy,
                                                                                            self.skips)
    
    def _train_epoch(self, list sentences, list tags):
        """
        Trains for one epoch with all examples.
        """
        self.train_hits = 0
        self.error = 0
        self.total_items = 0
        self.skips = 0
        
        # shuffle data
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        
        for sent, sent_tags in izip(sentences, tags):
            self._tag_sentence(sent, True, sent_tags)
    
    def _corrections(self, np.ndarray[INT_t, ndim=2]  window, 
                     np.ndarray[FLOAT_t, ndim=1] scores, int tag):
        """
        Performs corrections in the network after a pattern
        has been seen. It calls backpropagation, weights 
        and features ajustments.
        """
        # find the logadd
        cdef np.ndarray exponentials = np.exp(scores)
        cdef float exp_sum = np.sum(exponentials)
        cdef float logadd = math.log(exp_sum)
        
        # update information about training
        self.total_items += 1
        error = logadd - scores[tag]
        self.error += error
        if scores.argmax() == tag:
            self.train_hits += 1
            
        # if the error is too low, don't bother training (saves time and avoids
        # overfitting). An error of 0.01 means a log-prob of -0.01 for the right
        # tag, i.e., more than 99% probability
        # error 0.69 -> 50% probability for right tag (minimal threshold)
        # error 0.22 -> 80%
        # error 0.1  -> 90% 
        if error <= 0.01:
            self.skips += 1
            return
        
        # the gradient at each output neuron i is given by:
        # exp(i) / exp_sum, for wrong labels
        # (exp(i) / exp_sum) - 1, for the right label 
        cdef np.ndarray[FLOAT_t, ndim=1] output_gradients = - exponentials / exp_sum
        output_gradients[tag] += 1
        output_gradients *= self.learning_rate
        
        
        """
        Backpropagate the error gradient.
        """
        # find the hidden gradients by backpropagating the output
        # gradients and multiplying the derivative
        cdef np.ndarray[FLOAT_t, ndim=1] hidden_gradients = output_gradients.dot(self.output_weights)
        
        # the derivative of tanh(x) is 1 - tanh^2(x)
        cdef np.ndarray derivatives = 1 - self.hidden_values ** 2
        hidden_gradients *= derivatives
        
        # backpropagate to input layer (in order to adjust features)
        # since no function is applied to the feature values, no derivative is needed
        # (or you can see it as f(x) = x --> f'(x) = 1)
        cdef np.ndarray[FLOAT_t, ndim=1] input_gradients = hidden_gradients.dot(self.hidden_weights)
        
        """
        Adjust the weights of the neural network.
        """
        #NOTE: this is the function which consumes most CPU time
        # because of arithmetic operations
        
        # adjust weights from input to hidden layer
        # repeat the gradients so we get a copy of all of them for each input neuron
        cdef np.ndarray[FLOAT_t, ndim=2] delta_matrix
        
        delta_matrix = np.tile(hidden_gradients, (self.input_size, 1)).T
        delta_matrix *= self.input_values
        self.hidden_weights += delta_matrix
        self.hidden_bias += hidden_gradients
        
        # adjust weights from hidden to output layer
        delta_matrix = np.tile(output_gradients, (self.hidden_size, 1)).T
        delta_matrix *= self.hidden_values
        self.output_weights += delta_matrix
        self.output_bias += output_gradients
        
        """
        Adjust the features indexed by the input window.
        """
        # the deltas that will be applied to feature tables
        # they are in the same sequence as the network receives them, i.e.,
        # [token1-table1][token1-table2][token2-table1][token2-table2] (...)
        cdef np.ndarray[FLOAT_t, ndim=1] deltas = input_gradients * self.input_values 
        
        # this tracks where the deltas for the next table begins
        # (used for efficiency reasons)
        cdef int start_from = 0
        cdef np.ndarray[FLOAT_t, ndim=2] table
        cdef np.ndarray[INT_t, ndim=1] token
        cdef num_features
        cdef int index
        cdef int num_table
        cdef list feature_tables = self.feature_tables 
        
        for token in window:
            
            for num_table, index in enumerate(token):
                
                table = feature_tables[num_table]
                num_features = table.shape[1]
                table[index] += deltas[start_from:start_from + num_features]
                start_from += num_features
                
    def save(self, filename):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, padding and 
        distance tables, but not other feature tables.
        """
        np.savez(filename, hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias,
                 word_window_size=self.word_window_size, 
                 input_size=self.input_size, hidden_size=self.hidden_size,
                 output_size=self.output_size, padding_left=self.padding_left,
                 padding_right=self.padding_right)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding and 
        distance tables, but not other feature tables.
        """
        data = np.load(filename)
        
        # cython classes don't have the __dict__ attribute
        # so we can't do an elegant self.__dict__.update(data)
        hidden_weights = data['hidden_weights']
        hidden_bias = data['hidden_bias']
        output_weights = data['output_weights']
        output_bias = data['output_bias']
        
        word_window_size = data['word_window_size']
        input_size = data['input_size']
        hidden_size = data['hidden_size']
        output_size = data['output_size']
        
        nn = Network(word_window_size, input_size, hidden_size, output_size,
                     hidden_weights, hidden_bias, output_weights, output_bias)
        
        nn.padding_left = data['padding_left']
        nn.padding_right = data['padding_right']
        nn.pre_padding = np.array((nn.word_window_size / 2) * [nn.padding_left])
        nn.pos_padding = np.array((nn.word_window_size / 2) * [nn.padding_right])
        
        return nn
        
# include the file for the convolutional network
# this comes here after the Network class has already been defined
include "nlpnetconv.pyx"

