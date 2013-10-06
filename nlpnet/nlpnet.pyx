# -*- coding: utf-8 -*-

"""
A neural network for NLP tagging tasks.
It employs feature tables to store feature vectors for each token.
"""

import numpy as np
cimport numpy as np
cimport cython
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
    
    # transitions
    cdef public float learning_rate_trans
    cdef public np.ndarray transitions
    
    # the score for a given path
    cdef readonly float answer_score
    
    # gradients
    cdef readonly np.ndarray net_gradients, trans_gradients
    cdef readonly np.ndarray input_sent_values, hidden_sent_values
    
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
        
        self.transitions = np.zeros((self.output_size + 1, self.output_size))
        
        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias
    
    def description(self):
        """
        Returns a description of the network.
        """
        table_dims = [str(t.shape[1]) for t in self.feature_tables]
        table_dims =  ', '.join(table_dims)
        
        desc = """
Word window size: %d
Feature table sizes: %s
Input layer size: %d
Hidden layer size: %d
Output size: %d
""" % (self.word_window_size, table_dims, self.input_size, self.hidden_size, self.output_size)
        
        return desc
    
    
    def run(self, np.ndarray indices):
        """
        Runs the network for a given input. 
        :param indices: a 2-dim np array of indices to the feature tables.
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
    
    def tag_sentence(self, np.ndarray sentence, logprob=False):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        :param sentence: a 2-dim numpy array, where each item
        encodes a token.
        :param logprob: a boolean indicating whether to return the 
        log-probability for each answer or not.
        """
        return self._tag_sentence(sentence, train=False, logprob=logprob)
    
    def _tag_sentence(self, np.ndarray sentence, bool train=False, tags=None, logprob=False):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        :param sentence: a 2-dim numpy array, where each item
        encodes a token.
        :param train: if True, perform weight and feature correction.
        :param tags: the correct tags (needed when training)
        """
        cdef np.ndarray answer
        cdef np.ndarray scores = np.empty((len(sentence), self.output_size))
        
        if train:
            self.input_sent_values = np.empty((len(sentence), self.input_size))
            self.hidden_sent_values = np.empty((len(sentence), self.hidden_size))
        
        # add padding to the sentence
        cdef np.ndarray padded_sentence = np.vstack((self.pre_padding,
                                                     sentence,
                                                     self.pos_padding))

        # get the first window
        cdef np.ndarray window = padded_sentence[:self.word_window_size]
        cdef np.ndarray result = self.run(window)
        scores[0] = result
        if train:
            self.input_sent_values[0] = self.input_values
            self.hidden_sent_values[0] = self.hidden_values
        
        cdef object iter_tags
        if train:
            iter_tags = iter(tags)
        
        # run for the rest of the windows in the sentence
        cdef np.ndarray element
        for i, element in enumerate(padded_sentence[self.word_window_size:], 1):
            window = np.vstack((window[1:], element))
            result = self.run(window)
            scores[i] = result
            if train:
                self.input_sent_values[i] = self.input_values
                self.hidden_sent_values[i] = self.hidden_values 
        
        answer = self._viterbi(scores)
        if train:
            self._evaluate(answer, tags)
            if self._calculate_gradients_all_tokens(tags, scores):
                self._backpropagate(sentence)
                if self.transitions is not None: self._adjust_transitions()
         
        if logprob:
            if self.transitions is not None:
                all_scores = self._calculate_all_scores(scores)
                last_token = len(sentence) - 1
                logadd = np.log(np.sum(np.exp(all_scores[last_token])))
                confidence = self.answer_score - logadd
            
            else:
                confidence = np.prod(scores.max(1))
            
            answer = (answer, confidence)
         
        return answer
    
    def _evaluate(self, answer, tags):
        """
        Evaluates the network performance, updating its hits count.
        """
        for net_tag, gold_tag in zip(answer, tags):
            if net_tag == gold_tag:
                self.train_hits += 1
        self.total_items += len(tags)
    
    def _calculate_all_scores(self, scores):
        """
        Calculates a matrix with the scores for all possible paths at all given
        points (tokens).
        In the returning matrix, all_scores[i][j] means the sum of all scores 
        ending in token i with tag j
        """
        # logadd for first token. the transition score of the starting tag must be used.
        # it turns out that logadd = log(exp(score)) = score
        # (use long double because taking exp's leads to very very big numbers)
        scores = np.longdouble(scores)
        scores[0] += self.transitions[-1]
        
        # logadd for the following tokens
        transitions = self.transitions[:-1].T
        for token, _ in enumerate(scores[1:], 1):
            logadd = np.log(np.sum(np.exp(scores[token - 1] + transitions), 1))
            scores[token] += logadd
            
        return scores
    
    def _calculate_gradients_all_tokens(self, tags, scores):
        """
        Calculates the output and transition deltas for each token.
        The aim is to minimize the cost:
        logadd(score for all possible paths) - score(correct path)
        :returns: if True, normal gradient calculation was performed.
        If False, the error was too low and weight correction should be
        skipped.
        """
        cdef np.ndarray all_scores 
        
        correct_path_score = 0
        last_tag = self.output_size
        for tag, net_scores in izip(tags, scores):
            trans = 0 if self.transitions is None else self.transitions[last_tag, tag]
            correct_path_score += trans + net_scores[tag]
            last_tag = tag 
        
        all_scores = self._calculate_all_scores(scores)
        error = np.log(np.sum(np.exp(all_scores[-1]))) - correct_path_score
        self.error += error
        if error <= 0.01:
            self.skips += 1
            return False
        
        # initialize gradients
        self.net_gradients = np.zeros_like(scores, np.float)
        self.trans_gradients = np.zeros_like(self.transitions, np.float)
        
        # things get nasty from here
        # refer to the papers to understand what exactly is going on
        
        # compute the gradients for the last token
        exponentials = np.exp(all_scores[-1])
        exp_sum = np.sum(exponentials)
        self.net_gradients[-1] = -exponentials / exp_sum
        
        transitions_t = 0 if self.transitions is None else self.transitions[:-1].T
        
        # now compute the gradients for the other tokens, from last to first
        for token in range(len(scores) - 2, -1, -1):
            
            # matrix with the exponentials which will be used to find the gradients
            # sum the scores for all paths ending with each tag in token "token"
            # with the transitions from this tag to the next
            exp_matrix = np.exp(all_scores[token] + transitions_t).T
            
            # the sums of exps, used to calculate the softmax
            # sum the exponentials by column
            denominators = exp_matrix.sum(0)
            
            # softmax is the division of an exponential by the sum of all exponentials
            # (yields a probability)
            softmax = exp_matrix / denominators
            
            # multiply each value in the softmax by the gradient at the next tag
            grad_times_softmax = self.net_gradients[token + 1] * softmax
            self.trans_gradients[:-1, :]  += grad_times_softmax
            
            # sum all transition gradients by line to find the network gradients
            self.net_gradients[token] = np.sum(grad_times_softmax, 1)
        
        # find the gradients for the starting transition
        # there is only one possibility to come from, which is the sentence start
        self.trans_gradients[-1] = self.net_gradients[0]
        
        # now, add +1 to the correct path
        last_tag = self.output_size
        for token, tag in enumerate(tags):
            self.net_gradients[token][tag] += 1
            if self.transitions is not None:
                self.trans_gradients[last_tag][tag] += 1
            last_tag = tag
        
        return True
        
    def _adjust_transitions(self):
        """
        Adjusts the transition scores table with the calculated gradients.
        """
        self.transitions += self.trans_gradients * self.learning_rate_trans
    
    @cython.boundscheck(False)
    def _viterbi(self, np.ndarray[FLOAT_t, ndim=2] scores, bool allow_repeats=True):
        """
        Performs a Viterbi search over the scores for each tag using
        the transitions matrix. If a matrix wasn't supplied, 
        it will return the tags with the highest scores individually.
        """
        # pretty straightforward
        if self.transitions is None or len(scores) == 1:
            return scores.argmax(1)
            
        path_scores = np.empty_like(scores)
        path_backtrack = np.empty_like(scores, np.int)
        
        # now the actual Viterbi algorithm
        # first, get the scores for each tag at token 0
        # the last row of the transitions table has the scores for the first tag
        path_scores[0] = scores[0] + self.transitions[-1]
        
        for i, token in enumerate(scores[1:], 1):
            
            # each line contains the score until each tag t plus the transition to each other tag t'
            prev_score_and_trans = (path_scores[i - 1] + self.transitions[:-1].T).T
            
            # find the previous tag that yielded the max score
            path_backtrack[i] = prev_score_and_trans.argmax(0)
            path_scores[i] = prev_score_and_trans[path_backtrack[i], 
                                                  np.arange(self.output_size)] + scores[i]
            
        # now find the maximum score for the last token and follow the backtrack
        answer = np.empty(len(scores), dtype=np.int)
        answer[-1] = path_scores[-1].argmax()
        self.answer_score = path_scores[-1][answer[-1]]
        previous_tag = path_backtrack[-1][answer[-1]]
        
        for i in range(len(scores) - 2, 0, -1):
            answer[i] = previous_tag
            previous_tag = path_backtrack[i][previous_tag]
        
        answer[0] = previous_tag
        return answer
    
    def train(self, list sentences, list tags, 
              int epochs, int epochs_between_reports=0,
              float desired_accuracy=0):
        """
        Trains the network to tag sentences.
        :param sentences: a list of 2-dim numpy arrays, where each item
        encodes a sentence. Each item in a sentence has the 
        indices to its features.
        :param tags: a list of 1-dim numpy arrays, where each item has
        the tags of the sentences.
        :param epochs: number of training epochs
        :param epochs_between_reports: number of epochs to wait between
        reports about the training performance. 0 means no reports.
        :param desired_accuracy: training stops if the desired accuracy
        is reached. Ignored if 0.
        """
        print "Training for up to %d epochs" % epochs
        last_accuracy = 0
        last_error = np.Infinity 
        
        for i in range(epochs):
            self._train_epoch(sentences, tags)
            
            self.accuracy = float(self.train_hits) / self.total_items
            
            if (epochs_between_reports > 0 and i % epochs_between_reports == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.error > last_error):
                
                self._print_epoch_report(i + 1)
                
                if self.accuracy >= desired_accuracy > 0:
                    break
                
                if self.accuracy < last_accuracy and self.error > last_error:
                    # accuracy is falling, the network is probably diverging
                    break
            
            last_accuracy = self.accuracy
            last_error = self.error
        
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
            try:
                self._tag_sentence(sent, True, sent_tags)
            except FloatingPointError:
                # just ignore the sentence in case of an overflow
                continue
    
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
        
    def _backpropagate(self, sentence):
        """
        Backpropagate the error gradient.
        """
        # find the hidden gradients by backpropagating the output
        # gradients and multiplying the derivative
        cdef np.ndarray[FLOAT_t, ndim=2] hidden_gradients = self.net_gradients.dot(self.output_weights)
        
        # the derivative of tanh(x) is 1 - tanh^2(x)
        cdef np.ndarray derivatives = 1 - self.hidden_sent_values ** 2
        hidden_gradients *= derivatives
        
        # backpropagate to input layer (in order to adjust features)
        # since no function is applied to the feature values, no derivative is needed
        # (or you can see it as f(x) = x --> f'(x) = 1)
        cdef np.ndarray[FLOAT_t, ndim=2] input_gradients = hidden_gradients.dot(self.hidden_weights)
        
        """
        Adjust the weights of the neural network.
        """
        # tensor[i, j, k] means the gradient for tag i at token j to be multiplied
        # by the value from the k-th hidden neuron (note that the tensor was transposed)
        cdef np.ndarray[FLOAT_t, ndim=3] grad_tensor
        
        # adjust weights from input to hidden layer
        grad_tensor = np.tile(hidden_gradients, [self.input_size, 1, 1]).T
        grad_tensor *= self.input_sent_values
        deltas = grad_tensor.sum(1) * self.learning_rate
        self.hidden_weights += deltas
        self.hidden_bias += hidden_gradients.sum(0) * self.learning_rate
        
        # adjust weights from hidden to output layer
        grad_tensor = np.tile(self.net_gradients, [self.hidden_size, 1, 1]).T
        grad_tensor *= self.hidden_sent_values
        deltas = grad_tensor.sum(1) * self.learning_rate
        self.output_weights += deltas
        self.output_bias += self.net_gradients.sum(0) * self.learning_rate
        
        """
        Adjust the features indexed by the input window.
        """
        # the deltas that will be applied to feature tables
        # they are in the same sequence as the network receives them, i.e.,
        # [token1-table1][token1-table2][token2-table1][token2-table2] (...)
        input_deltas = input_gradients * self.input_sent_values * self.learning_rate_features
        
        # this tracks where the deltas for the next table begins
        # (used for efficiency reasons)
        cdef int start_from = 0
        cdef np.ndarray[FLOAT_t, ndim=2] table
        cdef np.ndarray[INT_t, ndim=1] token
        cdef num_features
        cdef int i, j
        
        padded_sentence = np.concatenate((self.pre_padding,
                                          sentence,
                                          self.pos_padding))
        
        for i in range(self.word_window_size):
            for j, table in enumerate(self.feature_tables):
                # this is the column for the i-th position in the window
                # regarding features from the j-th table
                table_deltas = input_deltas[:, start_from:start_from + table.shape[1]]
                start_from += table.shape[1]
                
                for token, deltas in zip(padded_sentence[i:], table_deltas):
                    table[token[j]] += deltas
        
    
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
                 padding_right=self.padding_right, transitions=self.transitions)
    
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
        
        if 'transitions' in data:
            transitions = data['transitions']
            if transitions.shape != ():
                nn.transitions = transitions 
        
        return nn
        
# include the file for the convolutional network
# this comes here after the Network class has already been defined
include "nlpnetconv.pyx"
include "nlpnetlm.pyx"

