# -*- coding: utf-8 -*-
#cython: embedsignature=True
#cython: profile=True

"""
A neural network for NLP tagging tasks.
It employs feature tables to store feature vectors for each token.
"""

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

from itertools import izip
import logging

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t
ctypedef np.double_t DOUBLE_t

# ----------------------------------------------------------------------
# Math functions

cdef logsumexp(np.ndarray a, axis=None):
    """Compute the log of the sum of exponentials of input elements.
    like: scipy.misc.logsumexp

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis over which the sum is taken. By default `axis` is None,
        and all elements are summed.

    Returns
    -------
    res : ndarray
        The result, ``np.log(np.sum(np.exp(a)))`` calculated in a numerically
        more stable way.
    """
    if axis is None:
        a = a.ravel()
    else:
        a = np.rollaxis(a, axis)
    a_max = a.max(axis=0)
    return np.log(np.sum(np.exp(a - a_max), axis=0)) + a_max

cdef hardtanh(np.ndarray[FLOAT_t, ndim=1] weights):
    """Hard hyperbolic tangent."""
    cdef np.ndarray out = np.copy(weights)
    inds_greater = weights > 1
    inds_lesser = weights < -1
    out[inds_greater] = 1
    out[inds_lesser] = -1
    
    return out

cdef hardtanhd(np.ndarray[FLOAT_t, ndim=2] weights):
    """derivative of hardtanh"""
    cdef np.ndarray out = np.zeros_like(weights)
    inds = np.logical_and(-1.0 <= weights, weights <= 1.0)
    out[inds] = 1.0
    
    return out

# ----------------------------------------------------------------------

cdef class Network:
    
    # sizes and learning rates
    cdef readonly int word_window_size, input_size, hidden_size, output_size
    cdef public float learning_rate, learning_rate_features
    cdef public float decay_factor
    cdef public bool use_learning_rate_decay
    
    # padding stuff
    cdef np.ndarray padding_left, padding_right
    cdef public np.ndarray pre_padding, pos_padding
    
    # weights, biases, calculated values
    cdef readonly np.ndarray hidden_weights, output_weights
    cdef readonly np.ndarray hidden_bias, output_bias
    cdef readonly np.ndarray input_values, hidden_values, layer2_values
    
    # feature tables
    cdef public list feature_tables
    
    # transitions
    cdef public float learning_rate_trans
    cdef public np.ndarray transitions
    
    # the score for a given path
    cdef readonly float answer_score
    
    # gradients
    cdef readonly np.ndarray net_gradients, trans_gradients
    cdef readonly np.ndarray input_sent_values, hidden_sent_values, layer2_sent_values
    
    # data for statistics during training. 
    cdef float error, accuracy, float_errors
    cdef int num_tokens, skips
        
    # file where the network is saved
    cdef public str network_filename
    
    # validation
    cdef list validation_sentences
    cdef list validation_tags

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

        # set the seed for replicability
        #np.random.seed(42)

        # SENNA: centered uniform distribution with variance = 1/sqrt(fanin)
        # variance = 1/12 interval ^ 2
        # interval = 3.46 / fanin ^ 1/4
        #high = 1.732 / np.power(input_size, 0.25) # SENNA: 0.416
        high = 2.38 / np.sqrt(input_size) # [Bottou-88]
        #high = 0.1              # Fonseca
        hidden_weights = np.random.uniform(-high, high, (hidden_size, input_size))
        hidden_bias = np.random.uniform(-high, high, (hidden_size))
        #high = 1.732 / np.power(hidden_size, 0.25) # SENNA
        high = 2.38 / np.sqrt(hidden_size) # [Bottou-88]
        #high = 0.1              # Fonseca
        output_weights = np.random.uniform(-high, high, (output_size, hidden_size))
        output_bias = np.random.uniform(-high, high, (output_size))
        
        high = 1.0
        # +1 is due for the initial transition
        transitions = np.random.uniform(-high, high, (output_size + 1, output_size))

        net = Network(word_window, input_size, hidden_size, output_size,
                      hidden_weights, hidden_bias, output_weights, output_bias,
                      transitions)
        net.feature_tables = feature_tables
        
        return net
        
    def __init__(self, word_window, input_size, hidden_size, output_size,
                 hidden_weights, hidden_bias, output_weights, output_bias,
                 transitions=None):
        """
        This function isn't expected to be directly called.
        Instead, use the classmethods load_from_file or 
        create_new.
        
        :param transitions: transition weights. If None uses
            Window Level Likelihood instead of Sentence Level Likelihood.
        """
        self.learning_rate = 0
        self.learning_rate_features = 0
        
        self.word_window_size = word_window
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # A_i_j score for jumping from tag i to j
        # A_0_i = transitions[-1]
        self.transitions = transitions

        self.hidden_weights = hidden_weights
        self.hidden_bias = hidden_bias
        self.output_weights = output_weights
        self.output_bias = output_bias
        
        self.validation_sentences = None
        self.validation_tags = None
        
        self.use_learning_rate_decay = False

    def description(self):
        """
        Returns a textual description of the network.
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
        
        :param indices: a 2-dim np array of indices into the feature tables.
            Each row represents a token through its indices into each feature table.
        """
        # find the actual input values concatenating the feature vectors
        # for each input token
        cdef np.ndarray input_data
        input_data = np.concatenate([table[index] 
                                     for token_indices in indices
                                     for index, table in izip(token_indices, 
                                                              self.feature_tables)
                                     ])

        # store the output in self for use in backpropagation
        self.input_values = input_data
        # (hidden_size, input_size) . input_size = hidden_size
        self.layer2_values = self.hidden_weights.dot(input_data) + self.hidden_bias
        self.hidden_values = hardtanh(self.layer2_values)

        return self.output_weights.dot(self.hidden_values) + self.output_bias

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
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        """
        scores = self._tag_sentence(sentence)
        # computes full score, combining ftheta and A (if SLL)
        return self._viterbi(scores)

    def _tag_sentence(self, np.ndarray sentence, tags=None):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        :param tags: the correct tags (needed when training)
        :return: a (len(sentence), output_size) array with the scores for all tokens
        """
        cdef np.ndarray answer
        # scores[t, i] = ftheta_i,t = score for i-th tag, t-th word
        cdef np.ndarray scores = np.empty((len(sentence), self.output_size))
        
        training = tags is not None
        if training:
            self.input_sent_values = np.empty((len(sentence), self.input_size))
            # layer2_values at each token in the correct path
            self.layer2_sent_values = np.empty((len(sentence), self.hidden_size))
            # hidden_values at each token in the correct path
            self.hidden_sent_values = np.empty((len(sentence), self.hidden_size))
        
        # add padding to the sentence
        cdef np.ndarray padded_sentence = np.concatenate((self.pre_padding,
                                                          sentence,
                                                          self.pos_padding))

        # run through all windows in the sentence
        for i in xrange(len(sentence)):
            window = padded_sentence[i: i+self.word_window_size]
            scores[i] = self.run(window)
            if training:
                self.input_sent_values[i] = self.input_values
                self.layer2_sent_values[i] = self.layer2_values
                self.hidden_sent_values[i] = self.hidden_values
        
        if training:
            if self._calculate_gradients_sll(tags, scores):
                self._backpropagate(sentence)

        return scores
    
    def _calculate_delta(self, scores):
        """
        Calculates a matrix with the scores for all possible paths at all given
        points (tokens).
        In the returned matrix, delta[i][j] means the sum of all scores 
        ending in token i with tag j (delta_i(j) in eq. 14 in the paper)
        """
        # logadd for first token. the transition score of the starting tag must be used.
        # it turns out that logadd = log(exp(score)) = score
        # (use long double because taking exp's leads to very very big numbers)
        # scores[t][k] = ftheta_k,t
        # No longer needed, since we use logsumexp
        #delta = np.longdouble(scores)
        delta = scores
        # transitions[len(sentence)] represents initial transition, A_0,i in paper (mispelled as A_i,0)
        # delta_0(k) = ftheta_k,0 + A_0,i
        delta[0] += self.transitions[-1]
        
        # logadd for the remaining tokens
        # delta_t(k) = ftheta_k,t + logadd_i(delta_t-1(i) + A_i,k)
        #            = ftheta_k,t + log(Sum_i(exp(delta_t-1(i) + A_i,k)))
        transitions = self.transitions[:-1].T # A_k,i
        for token in xrange(1, len(delta)):
            # sum by rows
            #logadd = np.log(np.sum(np.exp(delta[token - 1] + transitions), 1))
            logadd = logsumexp(delta[token - 1] + transitions, 1)
            delta[token] += logadd
            
        return delta

    @cython.boundscheck(False)
    def _calculate_gradients_sll(self, tags, scores):
        """
        Calculates the output and transition deltas for each token, using Sentence Level Likelihood.
        The aim is to minimize the cost:
        C(theta,A) = logadd(scores for all possible paths) - score(correct path)
        
        :returns: if True, normal gradient calculation was performed.
            If False, the error was too low and weight correction should be
            skipped.
        """
        cdef np.ndarray[DOUBLE_t, ndim=2] delta # (len(sentence), output_size)
        cdef np.ndarray[DOUBLE_t, ndim=2] delta_softmax # (output_size, output_size)
        
        # ftheta_i,t = network output for i-th tag, at t-th word
        # s = Sum_i(A_tags[i-1],tags[i] + ftheta_i,i), i < len(sentence)   (12)
        correct_path_score = 0
        last_tag = self.output_size
        for tag, net_scores in izip(tags, scores):
            trans = 0 if self.transitions is None else self.transitions[last_tag, tag]
            correct_path_score += trans + net_scores[tag]
            last_tag = tag 
        
        # delta[t] = delta_t in equation (14)
        delta = self._calculate_delta(scores)
        # logadd_i(delta_T(i)) = log(Sum_i(exp(delta_T(i))))
        # Sentence-level Log-Likelihood (SLL)
        # C(ftheta,A) = logadd_j(s(x, j, theta, A)) - score(correct path)
        #error = np.log(np.sum(np.exp(delta[-1]))) - correct_path_score
        error = logsumexp(delta[-1]) - correct_path_score
        self.error += error
        
        # if the error is too low, don't bother training (saves time and avoids
        # overfitting). An error of 0.01 means a log-prob of -0.01 for the right
        # tag, i.e., more than 99% probability
        # error 0.69 -> 50% probability for right tag (minimal threshold)
        # error 0.22 -> 80%
        # error 0.1  -> 90%
        if error <= 0.01:
            self.skips += 1
            return False
        
        # initialize gradients
        # dC / dftheta
        self.net_gradients = np.zeros((len(tags), self.output_size))
        # dC / dA
        self.trans_gradients = np.zeros_like(self.transitions, np.float)
        
        # things get nasty from here
        # refer to the papers to understand what exactly is going on
        
        # compute the gradients for the last token
        # dC_logadd / ddelta_T(i) = e(delta_T(i))/Sum_k(e(delta_T(k)))
        # Compute it using the log:
        # log(e(delta_T(i))/Sum_k(e(delta_T(k)))) =
        # log(e(delta_T(i))) - log(Sum_k(e(delta_T(k)))) =
        # delta_T(i) - logsumexp(delta_T(k))
        # dC_logadd / ddelta_T(i) = e(delta_T(i) - logsumexp(delta_T(k)))
        sumlogadd = logsumexp(delta[-1])
        # negative gradients
        self.net_gradients[-1] = -np.exp(delta[-1] - sumlogadd)

        transitions_t = 0 if self.transitions is None else self.transitions[:-1].T
        
        # delta[i][j]: sum of scores of all path that assign tag j to ith-token

        # now compute the gradients for the other tokens, from last to first
        for t in range(len(scores) - 2, -1, -1):
            
            # sum the scores for all paths ending with each tag i at token t
            # with the transitions from tag i to the next tag j
            # Obtained by transposing twice
            # [delta_t-1(i)+A_j,i]T
            path_scores = (delta[t] + transitions_t).T

            # normalize over all possible tag paths using a softmax,
            # computed using log.
            # the log of the sums of exps, summed by column
            log_sum_scores = logsumexp(path_scores, 0)
            
            # softmax is the division of an exponential by the sum of all exponentials
            # (yields a probability)
            # e(delta_t-1(i)+A_i,j) / Sum_k e(delta_t-1(k)+A_k,j)
            delta_softmax = np.exp(path_scores - log_sum_scores)

            # multiply each value in the softmax by the gradient at the next tag
            # dC_logadd / ddelta_t(i) * delta_softmax
            # Attardi: negative since net_gradients[t + 1] already negative
            grad_times_softmax = self.net_gradients[t + 1] * delta_softmax
            # dC / dA_i,j
            self.trans_gradients[:-1, :] += grad_times_softmax
            
            # sum all transition gradients by row to find the network gradients
            # Sum_j(dC_logadd / ddelta_t(j) * delta_softmax)
            # Attardi: negative since grad_times_softmax already negative
            self.net_gradients[t] = np.sum(grad_times_softmax, 1)

        # find the gradients for the starting transition
        # there is only one possibility to come from, which is the sentence start
        self.trans_gradients[-1] = self.net_gradients[0]
        
        # now, add +1 to the correct path
        last_tag = self.output_size
        for token, tag in enumerate(tags):
            self.net_gradients[token][tag] += 1 # negative gradient
            if self.transitions is not None:
                self.trans_gradients[last_tag][tag] += 1 # negative gradient
            last_tag = tag
        
        return True

    @cython.boundscheck(False)
    def _calculate_gradients_wll(self, tags, scores):
        """
        Calculates the output for each token, using Word Level Likelihood.
        The aim is to minimize the word-level log-likelihood:
        C(ftheta) = logadd_j(ftheta_j) - ftheta_y,
        where y is the sequence of correct tags
        
        :returns: if True, normal gradient calculation was performed.
            If False, the error was too low and weight correction should be
            skipped.
        """
        # compute the negative gradient with respect to ftheta
        # dC / dftheta_i = e(ftheta_i)/Sum_k(e(ftheta_k))
        exponentials = np.exp(scores)
        # FIXME: use logsumexp
        # ((len(sentence), self.output_size))
        self.net_gradients = -(exponentials.T / exponentials.sum(1)).T

        # correct path and its gradient
        correct_path_score = 0
        token = 0
        for tag, net_scores in izip(tags, scores):
            self.net_gradients[token][tag] += 1 # negative gradient
            token += 1
            correct_path_score += net_scores[tag]

        # C(ftheta) = logadd_j(ftheta_j) - score(correct path)
        #error = np.log(np.sum(np.exp(scores))) - correct_path_score
        error = logsumexp(scores) - correct_path_score
        # approximate
        #error = np.max(scores) - correct_path_score
        self.error += error

        return True

    @cython.boundscheck(False)
    def _viterbi(self, np.ndarray[FLOAT_t, ndim=2] scores):
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
        
        output_range = np.arange(self.output_size) # outside loop. Attardi
        transitions = self.transitions[:-1]        # idem

        cdef int i
        for i in xrange(1, len(scores)):
            
            # each line contains the score until each tag t plus the transition to each other tag t'
            prev_score_and_trans = (path_scores[i - 1] + transitions.T).T
            
            # find the previous tag that yielded the max score
            path_backtrack[i] = prev_score_and_trans.argmax(0)
            path_scores[i] = prev_score_and_trans[path_backtrack[i], 
                                                  output_range] + scores[i]
            
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
    
    def set_validation_data(self, list validation_sentences=None,
                            list validation_tags=None):
        """
        Sets the data to be used during validation. If this function is not
        called before training, the training data is used to measure performance.
        
        :param validation_sentences: sentences to be used in validation.
        :param validation_tags: tags for the validation sentences.
        """
        self.validation_sentences = validation_sentences
        self.validation_tags = validation_tags 
    
    def set_learning_rate_decay(self, float decay_factor=1.0):
        """
        Sets the network to use learning rate decay. 
        
        The learning rate at each iteration t is determined as:
        initial_rate / (1 + t * decay_factor)
        
        with t starting from 0
        """
        self.use_learning_rate_decay = True
        self.decay_factor = decay_factor
    
    def decrease_learning_rates(self, epoch):
        """
        Apply the learning rate decay, if the network was configured to use it.
        """
        if not self.use_learning_rate_decay or epoch == 0:
            return
        
        # multiplying the last rate by this adjustment is equivalent to
        # initial_rate / (1 + t * decay_factor)
        # and we don't need to store the initial rates
        factor = (1.0 + (epoch - 1) * self.decay_factor) / (1 + epoch * self.decay_factor)
        self.learning_rate *= factor
        self.learning_rate_features *= factor
        if self.transitions is not None:
            self.learning_rate_trans *= factor
    
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
        logger = logging.getLogger("Logger")
        logger.info("Training for up to %d epochs" % epochs)
        top_accuracy = 0
        last_accuracy = 0
        last_error = np.Infinity
        self.num_tokens = sum(len(sent) for sent in sentences)
        
        np.seterr(all='raise')
        if self.validation_sentences is None:
            self.validation_sentences = sentences
            self.validation_tags = tags
        
        for i in xrange(epochs):
            self.decrease_learning_rates(i)
            
            self._train_epoch(sentences, tags)
            self._validate()
            
            # normalize error
            self.error = self.error / self.num_tokens if self.num_tokens else np.Infinity
            
            # Attardi: save model
            if self.accuracy > top_accuracy:
                top_accuracy = self.accuracy
                self.save()
                logger.debug("Saved model")
            elif self.use_learning_rate_decay:
                # this iteration didn't bring improvements; load the last saved model
                # before continuing training with a lower rate
                self._load_parameters()
                        
            if (epochs_between_reports > 0 and i % epochs_between_reports == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.error > last_error):
                
                self._print_epoch_report(i + 1)
                
                if self.accuracy >= desired_accuracy > 0\
                        or (self.error > last_error and self.accuracy < last_accuracy):
                    break
                
            last_accuracy = self.accuracy
            last_error = self.error
        
        self.num_tokens = 0
            
    def _print_epoch_report(self, int num):
        """
        Reports the status of the network in the given training
        epoch, including error and accuracy.
        """
        logger = logging.getLogger("Logger")
        logger.info("%d epochs   Error: %f   Accuracy: %f   " \
            "%d corrections skipped   " \
            "learning rate: %f" % (num,
                                   self.error,
                                   self.accuracy,
                                   self.skips,
                                   self.learning_rate))
    
    def _train_epoch(self, list sentences, list tags):
        """
        Trains for one epoch with all examples.
        """
        self.error = 0
        self.skips = 0
        self.float_errors = 0
        
        # shuffle data
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        
        i = 0
        for sent, sent_tags in izip(sentences, tags):
            try:
                self._tag_sentence(sent, sent_tags)
            except FloatingPointError:
                # just ignore the sentence in case of an overflow
                self.float_errors += 1

    def _validate(self):
        """Perform validation on validation data and estimate accuracy"""
        hits = 0
        num_tokens = 0
        
        for sent, gold_tags in zip(self.validation_sentences, self.validation_tags):
            scores = self._tag_sentence(sent)
            answer = self._viterbi(scores)
            for pred_tag, gold_tag in izip(answer, gold_tags):
                if pred_tag == gold_tag:
                    hits += 1
            num_tokens += len(sent)
        
        self.accuracy = float(hits) / num_tokens

    def _backpropagate(self, sentence):
        """
        Backpropagate the gradients of the cost.
        """
        # f_1 = input_sent_values
        # f_2 = M_1 f_1 + b_2 = layer2_values
        # f_3 = hardTanh(f_2) = hidden_values
        # f_4 = M_2 f_3 + b_4

        # For l = 4..1 do:
        # dC / dtheta_l = df_l / dtheta_l dC / df_l		(19)
        # dC / df_l-1 = df_l / df_l-1 dC / df_l			(20)

        """
        Compute the gradients of the cost for each layer
        """
        # layer 4: output layer
        # dC / dW_4 = dC / df_4 f_3.T				(22)
        # (len, output_size).T (len, hidden_size) = (output_size, hidden_size)
        cdef np.ndarray[FLOAT_t, ndim=2] output_gradients
        output_gradients = self.net_gradients.T.dot(self.hidden_sent_values)

        # dC / db_4 = dC / df_4					(22)
        # (output_size) += ((len(sentence), output_size))
        # sum by column, i.e. all changes through the sentence
        output_bias_gradients = self.net_gradients.sum(0)

        # dC / df_3 = M_2.T dC / df_4				(23)
        #  (len, output_size) (output_size, hidden_size) = (len, hidden_size)
        dCdf_3 = self.net_gradients.dot(self.output_weights)

        # layer 3: HardTanh layer
        # no weights to adjust

        # dC / df_2 = hardtanhd(f_2) * dC / df_3
        # (len, hidden_size) (len, hidden_size)
        # FIXME: this goes quickly to 0.
        dCdf_2 = hardtanhd(self.layer2_sent_values) * dCdf_3

        # df_2 / df_1 = M_1

        # layer 2: linear layer
        # dC / dW_2 = dC / df_2 f_1.T				(22)
        cdef np.ndarray[FLOAT_t, ndim=2] hidden_gradients
        # (len, hidden_size).T (len, input_size) = (hidden_size, input_size)
        hidden_gradients = dCdf_2.T.dot(self.input_sent_values)

        # dC / db_2 = dC / df_2					(22)
        # sum by column contribution by each token
        hidden_bias_gradients = dCdf_2.sum(0)

        # dC / df_1 = M_1.T dC / df_2
        cdef np.ndarray[FLOAT_t, ndim=2] input_gradients
        # (len, hidden_size) (hidden_size, input_size) = (len, input_size)
        input_gradients = dCdf_2.dot(self.hidden_weights)

        """
        Adjust the weights. 
        """
        self.output_weights += output_gradients * self.learning_rate
        self.output_bias += output_bias_gradients * self.learning_rate
        self.hidden_weights += hidden_gradients * self.learning_rate
        self.hidden_bias += hidden_bias_gradients * self.learning_rate

        """
        Adjust the features indexed by the input window.
        """
        # the deltas that will be applied to the feature tables
        # they are in the same sequence as the network receives them, i.e.
        # [token1-table1][token1-table2][token2-table1][token2-table2] (...)
        # input_size = num features * window (e.g. 60 * 5). Attardi
        cdef np.ndarray[FLOAT_t, ndim=2] input_deltas
        # (len, input_size)
        input_deltas = input_gradients * self.learning_rate_features
        
        padded_sentence = np.concatenate((self.pre_padding,
                                          sentence,
                                          self.pos_padding))
        
        cdef np.ndarray[INT_t, ndim=1] features
        cdef np.ndarray[FLOAT_t, ndim=2] table
        cdef int start, end, t
        cdef int i, j

        for i, w_deltas in enumerate(input_deltas):
            # for each window (w_deltas: 300, features: 5)
            # this tracks where the deltas for the next table begins
            start = 0
            for features in padded_sentence[i:i+self.word_window_size]:
                # select the columns for each feature_tables (t: 3)
                for t, table in enumerate(self.feature_tables):
                    end = start + table.shape[1]
                    table[features[t]] += w_deltas[start:end]
                    start = end

        # Adjusts the transition scores table with the calculated gradients.
        if self.transitions is not None:
            self.transitions += self.trans_gradients * self.learning_rate_trans

    def _load_parameters(self):
        """
        Loads weights, feature tables and transition tables previously saved.
        """
        data = np.load(self.network_filename)
        self.hidden_weights = data['hidden_weights']
        self.hidden_bias = data['hidden_bias']
        self.output_weights = data['output_weights']
        self.output_bias = data['output_bias']
        self.feature_tables = data['feature_tables']
        
        if 'transitions' in data:
            self.transitions = data['transitions']
        else:
            self.transitions = None
    
    def save(self):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, padding, 
        distance tables and other feature tables.
        """
        np.savez(self.network_filename, hidden_weights=self.hidden_weights,
                 output_weights=self.output_weights,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias,
                 word_window_size=self.word_window_size, 
                 input_size=self.input_size, hidden_size=self.hidden_size,
                 output_size=self.output_size, padding_left=self.padding_left,
                 padding_right=self.padding_right, transitions=self.transitions,
                 feature_tables=self.feature_tables)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding, 
        distance tables and other feature tables.
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
        if 'transitions' in data:
            transitions = data['transitions']
        else:
            transitions = None

        nn = Network(word_window_size, input_size, hidden_size, output_size,
                     hidden_weights, hidden_bias, output_weights, output_bias,
                     transitions)
        
        nn.padding_left = data['padding_left']
        nn.padding_right = data['padding_right']
        nn.pre_padding = np.array((nn.word_window_size / 2) * [nn.padding_left])
        nn.pos_padding = np.array((nn.word_window_size / 2) * [nn.padding_right])
        nn.feature_tables = list(data['feature_tables'])
        nn.network_filename = filename
        
        return nn
        
# include the files for other networks
# this comes here after the Network class has already been defined
include "networkconv.pyx"
include "networkdependency.pyx"

