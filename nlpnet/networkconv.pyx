# -*- coding: utf-8 -*-

"""
A convolutional neural network for NLP tagging tasks like SRL.
It employs feature tables to store feature vectors for each token.
"""

import numpy as np
cimport numpy as np

cdef class ConvolutionalNetwork(Network):
    
    # transition and distance feature tables
    cdef public np.ndarray target_dist_table, pred_dist_table
    cdef readonly np.ndarray target_dist_weights, pred_dist_weights
    cdef readonly int target_dist_offset, pred_dist_offset
    cdef readonly np.ndarray target_dist_lookup, pred_dist_lookup
    cdef readonly np.ndarray target_convolution_lookup, pred_convolution_lookup
    cdef readonly np.ndarray target_dist_deltas, pred_dist_deltas
    cdef readonly np.ndarray historical_target_gradients, historical_pred_gradients
    cdef readonly np.ndarray historical_target_weight_gradients
    cdef readonly np.ndarray historical_pred_weight_gradients
    
    # the second hidden layer
    cdef readonly int hidden2_size
    cdef readonly np.ndarray hidden2_weights, hidden2_bias
    cdef readonly np.ndarray hidden2_values
    cdef readonly np.ndarray hidden2_before_activation, hidden_before_activation
    cdef readonly np.ndarray historical_hidden2_gradients
    cdef readonly np.ndarray historical_hidden2_bias_gradients
    
    # lookup of convolution values (the same for each sentence, used to save time)
    cdef np.ndarray convolution_lookup
        
    # maximum convolution indices
    cdef readonly np.ndarray max_indices
    
    # number of targets (all tokens in a sentence or the provided arguments)
    # and variables for argument classifying
    cdef int num_targets
    cdef bool only_classify
    
    # for faster access 
    cdef int half_window
    
    # the convolution gradients 
    cdef np.ndarray hidden_gradients, hidden2_gradients
    cdef np.ndarray input_deltas
    
    # keeping statistics
    cdef int num_sentences
    
    # validation
    cdef list validation_predicates, validation_arguments
    
    @classmethod
    def create_new(cls, feature_tables, target_dist_table, pred_dist_table, 
                   int word_window, int hidden1_size, int hidden2_size, int output_size):
        """Creates a new convolutional neural network."""
        # sum the number of features in all tables except for distance 
        cdef int input_size = sum(table.shape[1] for table in feature_tables)
        input_size *= word_window
        
        dist_features_per_token = target_dist_table.shape[1] + pred_dist_table.shape[1]
        input_size_with_distance = input_size + (word_window * dist_features_per_token)
        
        # creates the weight matrices
        high = 2.38 / np.sqrt(input_size_with_distance) # [Bottou-88]
        hidden_weights = np.random.uniform(-high, high, (hidden1_size, input_size))
        
        num_dist_features = word_window * target_dist_table.shape[1]
        target_dist_weights = np.random.uniform(-high, high, (num_dist_features, hidden1_size))
        num_dist_features = word_window * pred_dist_table.shape[1]
        pred_dist_weights = np.random.uniform(-high, high, (num_dist_features, hidden1_size))
        
        high = 2.38 / np.sqrt(hidden1_size)
        hidden_bias = np.random.uniform(-high, high, hidden1_size)
        
        if hidden2_size > 0:
            hidden2_weights = np.random.uniform(-high, high, (hidden2_size, hidden1_size))
            high = 2.38 / np.sqrt(hidden2_size)
            hidden2_bias = np.random.uniform(-high, high, hidden2_size)
            output_dim = (output_size, hidden2_size)
        else:
            hidden2_weights = None
            hidden2_bias = None
            output_dim = (output_size, hidden1_size)
        
        high = 2.38 / np.sqrt(output_dim[1])
        output_weights = np.random.uniform(-high, high, output_dim)
        high = 2.38 / np.sqrt(output_size)
        output_bias = np.random.uniform(-high, high, output_size)
        
        net = cls(word_window, input_size, hidden1_size, hidden2_size, 
                  output_size, hidden_weights, hidden_bias, 
                  target_dist_weights, pred_dist_weights,
                  hidden2_weights, hidden2_bias, 
                  output_weights, output_bias)
        net.feature_tables = feature_tables
        net.target_dist_table = target_dist_table
        net.pred_dist_table = pred_dist_table
        
        return net
    
    def description(self):
        """Returns a textual description of the network."""
        hidden2_size = 0 if self.hidden2_weights is None else self.hidden2_size
        table_dims = [str(t.shape[1]) for t in self.feature_tables]
        table_dims =  ', '.join(table_dims)
        
        dist_table_dims = '%d, %d' % (self.target_dist_table.shape[1], self.pred_dist_table.shape[1])
        
        desc = """
Word window size: %d
Feature table sizes: %s
Distance table sizes (target and predicate): %s
Input layer size: %d
Convolution layer size: %d 
Second hidden layer size: %d
Output size: %d
""" % (self.word_window_size, table_dims, dist_table_dims, self.input_size, self.hidden_size,
       hidden2_size, self.output_size)
        
        return desc
    
    
    def __init__(self, word_window, input_size, hidden1_size, hidden2_size,
                 output_size, hidden1_weights, hidden1_bias, target_dist_weights, 
                 pred_dist_weights, hidden2_weights, hidden2_bias, 
                 output_weights, output_bias):
        super(ConvolutionalNetwork, self).__init__(word_window, input_size, 
                                                   hidden1_size, output_size, 
                                                   hidden1_weights, hidden1_bias, 
                                                   output_weights, output_bias)
        self.half_window = word_window / 2
        self.features_per_token = self.input_size / word_window
        
        self.l2_factor = 0
        self.dropout = 0
        
        self.transitions = None
        self.target_dist_lookup = None
        self.pred_dist_lookup = None
        self.target_dist_weights = target_dist_weights
        self.pred_dist_weights = pred_dist_weights
        
        self.hidden2_size = hidden2_size
        self.hidden2_weights = hidden2_weights
        self.hidden2_bias = hidden2_bias
        
        self.validation_predicates = None
        self.validation_arguments = None
        
        self.use_learning_rate_decay = False
    
    def _generate_save_dict(self):
        """
        Generates a dictionary with all parameters saved by the model.
        It is directly used by the numpy savez function.
        """
        d = dict(hidden_weights=self.hidden_weights,
                 target_dist_table=self.target_dist_table,
                 pred_dist_table=self.pred_dist_table,
                 target_dist_weights=self.target_dist_weights,
                 pred_dist_weights=self.pred_dist_weights,
                 output_weights=self.output_weights,
                 transitions=self.transitions,
                 hidden_bias=self.hidden_bias, output_bias=self.output_bias,
                 word_window_size=self.word_window_size, 
                 input_size=self.input_size, hidden_size=self.hidden_size,
                 output_size=self.output_size, hidden2_size=self.hidden2_size,
                 hidden2_weights=self.hidden2_weights, hidden2_bias=self.hidden2_bias,
                 padding_left=self.padding_left, padding_right=self.padding_right,
                 feature_tables=self.feature_tables, dropout=self.dropout)
        return d
    
    def save(self):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, padding and 
        distance tables, and other feature tables.
        """
        data = self._generate_save_dict()
        np.savez(self.network_filename, **data)

    @classmethod
    def _load_from_file(cls, data, filename):
        """
        Internal method for setting data read from a npz file.
        """
        # cython classes don't have the __dict__ attribute
        # so we can't do an elegant self.__dict__.update(data)
        hidden_weights = data['hidden_weights']
        hidden_bias = data['hidden_bias']
        hidden2_weights = data['hidden2_weights']
        
        hidden2_bias = data['hidden2_bias']
        output_weights = data['output_weights']
        output_bias = data['output_bias']
        
        word_window = data['word_window_size']
        input_size = data['input_size']
        hidden_size = data['hidden_size']
        hidden2_size = data['hidden2_size']
        output_size = data['output_size']
        
        # numpy stores None as an array containing None and with empty shape
        if hidden2_weights.shape == (): 
            hidden2_weights = None
            hidden2_size = 0
            hidden2_bias = None
        
        nn = cls(word_window, input_size, hidden_size, hidden2_size, 
                 output_size, hidden_weights, hidden_bias, 
                 data['target_dist_weights'], data['pred_dist_weights'],
                 hidden2_weights, hidden2_bias, 
                 output_weights, output_bias)
        
        nn.target_dist_table = data['target_dist_table']
        nn.pred_dist_table = data['pred_dist_table']
        transitions = data['transitions']
        nn.transitions = transitions if transitions.shape != () else None
        nn.padding_left = data['padding_left']
        nn.padding_right = data['padding_right']
        nn.pre_padding = np.array(int(nn.word_window_size / 2) * [nn.padding_left])
        nn.pos_padding = np.array(int(nn.word_window_size / 2) * [nn.padding_right])
        nn.feature_tables = list(data['feature_tables'])
        nn.network_filename = filename
        
        if 'dropout' in data:
            nn.dropout = data['dropout']
        else:
            nn.dropout = 0
        
        return nn

    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding and 
        distance tables, and other feature tables.
        """
        data = np.load(filename, encoding='bytes')
        return cls._load_from_file(data, filename)
    
    def _load_parameters(self):
        """
        Loads weights, feature tables, distance tables and 
        transition tables previously saved.
        """
        data = np.load(self.network_filename, encoding='bytes')
        self.hidden_weights = data['hidden_weights']
        self.hidden_bias = data['hidden_bias']
        self.output_weights = data['output_weights']
        self.output_bias = data['output_bias']
        self.feature_tables = list(data['feature_tables'])
        self.target_dist_table = data['target_dist_table']
        self.pred_dist_table = data['pred_dist_table']
        
        # check if transitions isn't None (numpy saves everything as an array)
        if data['transitions'].shape != ():
            self.transitions = data['transitions']
        else:
            self.transitions = None
            
        # same for second hidden layer weights
        if data['hidden2_weights'].shape != ():
            self.hidden2_weights = data['hidden2_weights']
            self.hidden2_bias = data['hidden2_bias']
        else:
            self.hidden2_weights = None
        
    def set_validation_data(self, list validation_sentences,
                            list validation_predicates,
                            list validation_tags,
                            list validation_arguments=None):
        """
        Sets the data to be used in validation during training. If this function
        is not called before training, the training data itself is used to 
        measure the model's performance.
        """
        self.validation_sentences = validation_sentences
        self.validation_predicates = validation_predicates
        self.validation_tags = validation_tags
        self.validation_arguments = validation_arguments
    
    def train(self, list sentences, list predicates, list tags,  
              int epochs, int epochs_between_reports=0,
              float desired_accuracy=0, list arguments=None):
        """
        Trains the convolutional network. Refer to the basic Network
        train method for detailed explanation.
        
        :param predicates: a list of 1-dim numpy array
            indicating the indices of predicates in each sentence.
        :param arguments: (only for argument classifying) a list of 2-dim
            numpy arrays indicating the start and end of each argument. 
        """
        self.num_sentences = len(sentences)
        self.num_tokens = sum(len(sent) for sent in sentences)
        self.only_classify = arguments is not None
        
        logger = logging.getLogger("Logger")
        logger.info("Training for up to %d epochs" % epochs)
        last_accuracy = 0
        top_accuracy = 0
        last_error = np.Infinity
        self.historical_output_gradients = np.zeros_like(self.output_weights)
        self.historical_hidden_gradients = np.zeros_like(self.hidden_weights)
        self.historical_hidden2_gradients = np.zeros_like(self.hidden2_weights)
        self.historical_input_gradients = np.zeros(self.input_size)
        self.historical_target_weight_gradients = np.zeros_like(self.target_dist_weights)
        self.historical_pred_weight_gradients = np.zeros_like(self.pred_dist_weights)
        self.historical_trans_gradients = np.zeros_like(self.transitions)
        self.historical_hidden_bias_gradients = np.zeros_like(self.hidden_bias)
        self.historical_hidden2_bias_gradients = np.zeros_like(self.hidden2_bias)
        self.historical_output_bias_gradients = np.zeros_like(self.output_bias)
        
        if self.validation_sentences is None:
            self.set_validation_data(sentences, predicates, tags, arguments)
        
        for i in xrange(epochs):
#             self.decrease_learning_rates(i)
            self._train_epoch(sentences, predicates, tags, arguments)
            self._validate()
            
            # Attardi: save model
            if self.accuracy > top_accuracy:
                top_accuracy = self.accuracy
                self.save()
                logger.debug("Saved model")
#             elif self.use_learning_rate_decay:
#                 # this iteration didn't bring improvements; load the last saved model
#                 # before continuing training with a lower rate
#                 self._load_parameters()
            
            if (epochs_between_reports > 0 and i % epochs_between_reports == 0) \
                or self.accuracy >= desired_accuracy > 0 \
                or (self.accuracy < last_accuracy and self.error > last_error):
                
                self._print_epoch_report(i + 1)

                if self.accuracy >= desired_accuracy > 0\
                        or (self.accuracy < last_accuracy and self.error > last_error):
                    # accuracy is falling, the network is probably diverging
                    # or overfitting
                    break
            
            last_accuracy = self.accuracy
            last_error = self.error
        
        self.num_sentences = 0
        self.num_tokens = 0
        self._reset_counters()
    
    def _reset_counters(self):
        """
        Reset the performance statistics counters. They are updated during
        each epoch. 
        """
        self.error = 0
        self.skips = 0
        self.float_errors = 0
    
    def _shuffle_data(self, sentences, predicates, tags, arguments=None):
        """
        Shuffle the given training data in place.
        """
        # get the random number generator state in order to shuffle
        # sentences and their tags in the same order
        random_state = np.random.get_state()
        np.random.shuffle(sentences)
        np.random.set_state(random_state)
        np.random.shuffle(predicates)
        np.random.set_state(random_state)
        np.random.shuffle(tags)
        if arguments is not None:
            np.random.set_state(random_state)
            np.random.shuffle(arguments)
        
        
    def _train_epoch(self, sentences, predicates, tags, arguments):
        """Trains for one epoch with all examples."""
        
        self._reset_counters()
        self._shuffle_data(sentences, predicates, tags, arguments)
        if arguments is not None:
            i_args = iter(arguments)
        else:
            sent_args = None
        
        for sent, sent_preds, sent_tags in zip(sentences, predicates, tags):
            if arguments is not None:
                sent_args = i_args.next()
            
            try:
                self._tag_sentence(sent, sent_preds, sent_tags, sent_args)
            except FloatingPointError:
                # just ignore the sentence in case of an overflow
                self.float_errors += 1
    
    def tag_sentence(self, np.ndarray sentence, np.ndarray predicates, 
                     list arguments=None, bool logprob=False,
                     bool allow_repeats=True):
        """
        Runs the network for each element in the sentence and returns 
        the sequence of tags.
        
        :param sentence: a 2-dim numpy array, where each item encodes a token.
        :param predicates: a 1-dim numpy array, indicating the position
            of the predicates in the sentence
        :param logprob: a boolean indicating whether to return the 
            log-probability for each answer or not.
        :param allow_repeats: a boolean indicating whether to allow repeated
            argument classes (only for separate argument classification).
        """
        self.only_classify = arguments is not None
        return self._tag_sentence(sentence, predicates, argument_blocks=arguments, 
                                  logprob=logprob, allow_repeats=allow_repeats)
    
    cdef np.ndarray argument_distances(self, positions, argument):
        """
        Calculates the distance from each token in the sentence to the argument.
        """
        distances = positions.copy()
        
        # the ones before the argument
        lo = np.less(positions, argument[0])
        distances[lo] -= argument[0]
        
        # the ones after the argument
        hi = np.greater(positions, argument[1])
        distances[hi] -= argument[1]
        
        # the ones inside the argument
        distances[np.logical_not(hi | lo)] = 0
        
        return distances
    
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _sentence_convolution(self, sentence, predicate, argument_blocks=None, 
                              training=False):
        """
        Perform the convolution for a given predicate.
        
        :param sentence: a sequence of tokens, each represented as an array of 
            indices
        :param predicate: the index of the predicate in the sentence
        :param argument_blocks: (used only in SRL argument classification) the
            starting and end positions of all delimited arguments
        :return: the scores for all tokens with respect to the given predicate
        """        
        # store the values found by each convolution neuron here and then find the max
        cdef np.ndarray[FLOAT_t, ndim=2] convolution_values
        
        # a priori scores for all tokens
        cdef np.ndarray[FLOAT_t, ndim=2] scores
        
        # intermediate storage
        cdef np.ndarray[FLOAT_t, ndim=2] input_and_pred_dist_values
        
        self.num_targets = len(sentence) if argument_blocks is None else len(argument_blocks)
        
        # maximum values found by convolution
        self.hidden_values = np.zeros((self.num_targets, self.hidden_size))
        
        if training:
            # hidden sent values: results after tanh
            self.hidden_values = np.zeros((self.num_targets, self.hidden_size))
            self.max_indices = np.empty((self.num_targets, self.hidden_size), np.int)
                
        # predicate distances are the same across all targets
        pred_dist_indices = np.arange(len(sentence)) - predicate
        pred_dist_values = self.pred_convolution_lookup.take(pred_dist_indices + self.pred_dist_offset,
                                                             0, mode='clip')
        
        input_and_pred_dist_values = pred_dist_values + self.convolution_lookup
        
        for target in range(self.num_targets):
            # loop over targets and add the weighted distance features to each token
            # this is necessary for the convolution layer
            
            # distance features for each window
            # if we are classifying all tokens, pick the distance to the target
            # if we are classifying arguments, pick the distance to the closest boundary 
            # of the argument (beginning or end)
            if argument_blocks is None:
                target_dist_indices = np.arange(len(sentence)) - target
            else:
                argument = argument_blocks[target]
                target_dist_indices = self.argument_distances(np.arange(len(sentence)), argument)
            
            target_dist_values = self.target_convolution_lookup.take(target_dist_indices + self.target_dist_offset,
                                                                     0, mode='clip')

            convolution_values = target_dist_values + input_and_pred_dist_values
            
            # now, find the maximum values
            if training:
                self.max_indices[target] = convolution_values.argmax(0)
            self.hidden_values[target] = convolution_values.max(0)
        
        self._dropout(self.hidden_values, training)
        
        # apply the bias and proceed to the next layer
        self.hidden_values += self.hidden_bias
        
        if self.hidden2_weights is not None:
            self.hidden2_values = self.hidden_values.dot(self.hidden2_weights.T) + self.hidden2_bias
            
            # dropout
            self._dropout(self.hidden2_values, training)
            
            if training:
                self.hidden2_before_activation = self.hidden2_values.copy()
    
            hardtanh(self.hidden2_values, inplace=True)
        else:
            # apply non-linearity here
            if training:
                self.hidden_before_activation = self.hidden_values.copy()
            
            self.hidden2_values = self.hidden_values
            hardtanh(self.hidden_values, inplace=True)
            
        scores = self.hidden2_values.dot(self.output_weights.T) + self.output_bias
        
        return scores
    
    def _pre_tagging_setup(self, np.ndarray sentence, bool training):
        """
        Perform some initialization actions before the actual tagging.
        """
        if training:
            # this table will store the values of the neurons for each input token
            # they will be needed during weight adjustments
            self.input_sent_values = np.empty((len(sentence), self.input_size))
        
        # store the convolution values to save time
        self._create_convolution_lookup(sentence, training)
        
        if self.target_dist_lookup is None: self._create_target_lookup()
        if self.pred_dist_lookup is None: self._create_pred_lookup()
        
        if self.historical_target_gradients is None:
            self.historical_target_gradients = np.zeros_like(self.target_dist_lookup)
            
        if self.historical_pred_gradients is None:
            self.historical_pred_gradients = np.zeros_like(self.pred_dist_lookup)
        
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _tag_sentence(self, np.ndarray sentence, np.ndarray predicates, 
                      list tags=None, list argument_blocks=None, 
                      bool allow_repeats=True, bool logprob=False):
        """
        Runs the network for every predicate in the sentence.
        Refer to the Network class for more information.
        
        :param tags: this is a list rather than a numpy array because in
            argument classification, each predicate may have a differente number
            of arguments.
        :param argument_blocks: (used only in SRL argument classification) a list
            with the starting and end positions of all delimited arguments (one for 
            each predicate)
        :param predicates: a numpy array with the indices of the predicates in the sentence.
        """
        answer = []
        training = tags is not None
        self._pre_tagging_setup(sentence, training)
        cdef np.ndarray[FLOAT_t, ndim=2] token_scores
        
        for i, predicate in enumerate(predicates):
            pred_arguments = None if not self.only_classify else argument_blocks[i]
            
            token_scores = self._sentence_convolution(sentence, predicate, pred_arguments, training)
            pred_answer = self._viterbi(token_scores, allow_repeats)
        
            if training:
                pred_tags = tags[i]
                if self._calculate_gradients(pred_tags, token_scores):
                    self._backpropagate()
                    self._calculate_input_deltas(sentence, predicate, pred_arguments)
                    self._adjust_weights(predicate, pred_arguments)
                    self._adjust_features(sentence, predicate)
            
            if logprob:
                if self.only_classify:
                    raise NotImplementedError('Confidence measure not implemented for argument classifying')
                
                all_scores = self._calculate_all_scores(token_scores)
                last_token = len(sentence) - 1
                logadd = np.log(np.sum(np.exp(all_scores[last_token])))
                confidence = self.answer_score - logadd
                pred_answer = (pred_answer, confidence)
            
            answer.append(pred_answer)
        
        return answer
        
    def _validate(self):
        """
        Evaluates the network performance, updating its hits count.
        """
        # call it "item" instead of token because the same token may be counted
        # more than once (sentences with multiple predicates)
        num_items = 0
        hits = 0
        
        if self.validation_arguments is not None:
            i_args = iter(self.validation_arguments)
        else:
            sent_args = None
        
        for sent, sent_preds, sent_tags in zip(self.validation_sentences,
                                                self.validation_predicates, 
                                                self.validation_tags):
            if self.validation_arguments is not None:
                sent_args = i_args.next()
            
            answer = self._tag_sentence(sent, sent_preds, None, sent_args)
            for predicate_answer, predicate_tags in zip(answer, sent_tags):
                for net_tag, gold_tag in zip(predicate_answer, predicate_tags):
                    if net_tag == gold_tag:
                        hits += 1
                
                num_items += len(predicate_answer)
        
        self.accuracy = float(hits) / num_items
        # normalize error
        self.error /= num_items
    
    def _calculate_gradients(self, tags, scores):
        """Delegates the call to the appropriate function."""
        if self.only_classify:
            return self._calculate_gradients_classify(tags, scores)
        else:
            return self._calculate_gradients_sll(tags, scores)
    
    def _calculate_gradients_classify(self, tags, scores):
        """
        Calculates the output deltas for each target in a network that only 
        classifies predelimited arguments.
        The aim is to minimize the cost, for each argument:
        logadd(score for all possible tags) - score(correct tag)
        
        :returns: whether a correction is necessary or not.
        """
        self.net_gradients = np.zeros_like(scores, np.float)
        correction = False
        
        for i, tag_scores in enumerate(scores):
            tag = tags[i]
            
            exponentials = np.exp(tag_scores)
            exp_sum = np.sum(exponentials)
            logadd = np.log(exp_sum)
            
            # update the total error 
            error = logadd - tag_scores[tag]
            self.error += error
            
            # like the non-convolutional network, don't adjust weights if the error
            # is too low. An error of 0.01 means a log-prob of -0.01 for the right
            # tag, i.e., more than 99% probability
            if error <= 0.01:
                self.skips += 1
                continue
            
            correction = True
            self.net_gradients[i] = - exponentials / exp_sum
            self.net_gradients[i, tag] += 1
    
        return correction

    def _backpropagate(self):
        """Backpropagates the error gradient."""
        
        # this function only determines the gradients at each layer, without 
        # adjusting weights. This is done because the input features must 
        # be adjusted with the first weight matrix unchanged. 
        
        # gradient[i][j] has the gradient for token i at neuron j
        
        # derivative with respect to the non-linearity layer (tanh)
        dCd_tanh = self.net_gradients.dot(self.output_weights)
        
        if self.hidden2_weights is not None:
            # derivative with respect to the second hidden layer
            self.hidden2_gradients = dCd_tanh * hardtanhd(self.hidden2_before_activation)
            self.hidden_gradients = self.hidden2_gradients.dot(self.hidden2_weights)
        else:
            # the non-linearity appears right after the convolution max
            self.hidden_gradients = dCd_tanh * hardtanhd(self.hidden_before_activation)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _adjust_weights(self, predicate, arguments=None):
        """
        Adjusts the network weights after gradients have been calculated.
        """
        cdef int i
        cdef np.ndarray[FLOAT_t, ndim=1] gradients_t, bias_deltas
        cdef np.ndarray[FLOAT_t, ndim=2] last_values, deltas, grad_matrix, input_values
        cdef float epsilon = 1e-10
        
        last_values = self.hidden2_values if self.hidden2_weights is not None else self.hidden_values
        
        deltas = self.net_gradients.T.dot(last_values) 
        bias_deltas = self.net_gradients.sum(0)
        
        self.adagrad(deltas, self.historical_output_gradients)
        self.adagrad(bias_deltas, self.historical_output_bias_gradients)
        
        # L2 regularization
        deltas -= self.output_weights * self.l2_factor
        
        self.output_weights += deltas * self.learning_rate
        self.output_bias += bias_deltas * self.learning_rate
        self.output_weights= self.cap_norm(self.output_weights)
        
        if self.hidden2_weights is not None:
            deltas = self.hidden2_gradients.T.dot(self.hidden_values)
            bias_deltas = self.hidden2_gradients.sum(0)
            self.adagrad(deltas, self.historical_hidden2_gradients)
            self.adagrad(bias_deltas, self.historical_hidden2_bias_gradients)
            
            # L2
            deltas -= self.hidden2_weights * self.l2_factor
            
            self.hidden2_weights += deltas * self.learning_rate
            self.hidden2_bias += bias_deltas * self.learning_rate
            self.hidden2_weights= self.cap_norm(self.hidden2_weights)
        
        # now adjust weights from input to convolution. these will be trickier.
        # we need to know which input value to use in the delta formula
        
        # I tried vectorizing this loop but it got a bit slower, probably because
        # of the overhead in building matrices/tensors with the max indices
        for i, neuron_maxes in enumerate(self.max_indices):
            # i indicates the i-th target
              
            gradients_t = self.hidden_gradients[i]
            
            # table containing in each line the input values selected for each convolution neuron
            input_values = self.input_sent_values.take(neuron_maxes, 0)
            
            # stack the gradients to multiply all weights for a neuron
            grad_matrix = np.tile(gradients_t, [self.input_size, 1]).T
            deltas = grad_matrix * input_values
            self.adagrad(deltas, self.historical_hidden_gradients)
            
            # L2
            deltas -= self.hidden_weights * self.l2_factor
            
            self.hidden_weights += deltas * self.learning_rate
            
            # target distance weights
            # get the relative distance from each max token to its target
            if arguments is None:
                target_dists = neuron_maxes - i
            else:
                argument = arguments[i]
                target_dists = self.argument_distances(neuron_maxes, argument)
            
            dist_features = self.target_dist_lookup.take(target_dists + self.target_dist_offset, 
                                                         0, mode='clip')
            grad_matrix = np.tile(gradients_t, [self.target_dist_weights.shape[0], 1]).T
            
            deltas = (grad_matrix * dist_features).T
            self.adagrad(deltas, self.historical_target_weight_gradients)
            
            # L2
            deltas -= self.target_dist_weights * self.l2_factor
            
            self.target_dist_weights += deltas * self.learning_rate
            
            # predicate distance weights
            # get the distance from each max token to its predicate
            pred_dists = neuron_maxes - predicate
            dist_features = self.pred_dist_lookup.take(pred_dists + self.pred_dist_offset,
                                                       0, mode='clip')
            # try to recycle the grad_matrix if sizes match
            if self.target_dist_weights.shape[0] != self.pred_dist_weights.shape[0]: 
                grad_matrix = np.tile(gradients_t, [self.pred_dist_weights.shape[0], 1]).T
            
            deltas = (grad_matrix * dist_features).T
            self.adagrad(deltas, self.historical_pred_weight_gradients)
            
            # L2
            deltas -= self.pred_dist_weights * self.l2_factor
            
            self.pred_dist_weights += deltas * self.learning_rate
        
        bias_deltas = self.hidden_gradients.sum(0)
        self.adagrad(bias_deltas, self.historical_hidden_bias_gradients)
        self.hidden_bias += bias_deltas * self.learning_rate
        
        self.hidden_weights = self.cap_norm(self.hidden_weights)
        self.target_dist_weights = self.cap_norm(self.target_dist_weights)
        self.pred_dist_weights = self.cap_norm(self.pred_dist_weights)
        
        # Adjusts the transition scores table with the calculated gradients.
        if not self.only_classify and self.transitions is not None:
            self.historical_trans_gradients += self.trans_gradients ** 2
            deltas = self.trans_gradients / (epsilon + np.sqrt(self.historical_trans_gradients))
            self.transitions += self.learning_rate * deltas
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _calculate_input_deltas(self, np.ndarray sentence, int predicate, 
                                 object arguments=None):
        """
        Calculates the input deltas to be applied in the feature tables.
        """
        cdef np.ndarray[FLOAT_t, ndim=2] hidden_gradients, input_gradients
        cdef np.ndarray[FLOAT_t, ndim=2] target_dist_gradients, pred_dist_gradients
        cdef np.ndarray[FLOAT_t, ndim=1] gradients
        cdef np.ndarray[INT_t, ndim=1] convolution_max, target_dists
        
        # matrices accumulating gradients over each target
        input_gradients = np.zeros((len(sentence), self.hidden_size))
        target_dist_gradients = np.zeros((self.target_dist_lookup.shape[0], self.hidden_size))
        pred_dist_gradients = np.zeros((self.pred_dist_lookup.shape[0], self.hidden_size))
        
        # avoid multiplying by the learning rate multiple times
        hidden_gradients = self.hidden_gradients * self.learning_rate
        cdef np.ndarray[INT_t, ndim=1] column_numbers = np.arange(self.hidden_size)
        
        for target in range(self.num_targets):
            
            # array with the tokens that yielded the maximum value in each neuron
            # for this target
            convolution_max = self.max_indices[target]
            
            if not self.only_classify:
                target_dists = convolution_max - target
            else:
                argument = arguments[target]
                target_dists = self.argument_distances(convolution_max, argument)
            
            target_dists = np.clip(target_dists + self.target_dist_offset, 0,
                                   self.target_dist_lookup.shape[0] - 1)
            pred_dists = convolution_max - predicate
            pred_dists = np.clip(pred_dists + self.pred_dist_offset, 0,
                                 self.pred_dist_lookup.shape[0] - 1)
            
            gradients = hidden_gradients[target]
            
            # sparse matrix with gradients to be applied over the input
            # line i has the gradients for the i-th token in the sentence
            input_gradients[convolution_max, column_numbers] += gradients
            
            # distance deltas
            target_dist_gradients[target_dists, column_numbers] += gradients
            pred_dist_gradients[pred_dists, column_numbers] += gradients
        
        # (len(sent), hidden_size) . (hidden_size, input_size) = (len(sent), input_size)
        self.input_deltas = input_gradients.dot(self.hidden_weights)
        
        # (dist_lookup_size, hidden_size) . (hidden_size, num_dist_features) = (dist_lookup_size, num_dist_features)
        self.target_dist_deltas = target_dist_gradients.dot(self.target_dist_weights.T)
        self.pred_dist_deltas = pred_dist_gradients.dot(self.pred_dist_weights.T)
        
        # use adagrad on the embedding deltas
        # each matrix has a number of rows equal to the sentence length
        # the historical input gradients are stored in a vector;
        # the ones for distance tables are stored in matrices 
        squared_deltas = self.input_deltas ** 2
        self.historical_input_gradients += squared_deltas.sum(0)
        self.input_deltas /= np.sqrt(self.historical_input_gradients)
        
        self.historical_target_gradients += self.target_dist_deltas ** 2
        self.target_dist_deltas /= np.sqrt(self.historical_target_gradients)
        
        self.historical_pred_gradients += self.pred_dist_deltas ** 2
        self.pred_dist_deltas /= np.sqrt(self.historical_pred_gradients)            
        
    def _adjust_features(self, sentence, predicate):
        """Adjusts the features in all feature tables."""
        # compute each token in the window separately and
        # separate the feature deltas into tables
        start_from = 0
        dist_target_from = 0
        dist_pred_from = 0
        
        # number of times that the minimum and maximum distances are repeated
        # in the lookup distance tables
        pre_dist = self.word_window_size
        pos_dist = 1
        if self.word_window_size > 1:
            padded_sentence = np.concatenate((self.pre_padding,
                                              sentence,
                                              self.pos_padding))
        else:
            padded_sentence = sentence
        
        for i in range(self.word_window_size):
            
            for j, table in enumerate(self.feature_tables):
                # this is the column for the i-th position in the window
                # regarding features from the j-th table
                table_deltas = self.input_deltas[:, start_from:start_from + table.shape[1]]
                start_from += table.shape[1]
                
                for token, deltas in zip(padded_sentence[i:], table_deltas):
                    table[token[j]] += deltas
            
            dist_deltas = self.target_dist_deltas[:, dist_target_from : dist_target_from + self.target_dist_table.shape[1] ]
            pre_deltas = dist_deltas.take(np.arange(pre_dist), 0).sum(0)
            pos_deltas = dist_deltas.take(np.arange(-pos_dist, 0), 0).sum(0)
            self.target_dist_table[1:-1, :] += dist_deltas[pre_dist : -pos_dist]
            self.target_dist_table[0] += pre_deltas
            self.target_dist_table[-1] += pos_deltas 
            dist_target_from += self.target_dist_table.shape[1]
            
            dist_deltas = self.pred_dist_deltas[:, dist_pred_from : dist_pred_from + self.pred_dist_table.shape[1] ]
            pre_deltas = dist_deltas.take(np.arange(pre_dist), 0).sum(0)
            pos_deltas = dist_deltas.take(np.arange(-pos_dist, 0), 0).sum(0)
            self.pred_dist_table[1:-1, :] += dist_deltas[pre_dist : -pos_dist]
            self.pred_dist_table[0] += pre_deltas
            self.pred_dist_table[-1] += pos_deltas
            
            pre_dist -= 1
            pos_dist += 1
            dist_pred_from += self.pred_dist_table.shape[1]
            
        self._create_target_lookup()
        self._create_pred_lookup()
    
    @cython.boundscheck(False)
    def _viterbi(self, np.ndarray[FLOAT_t, ndim=2] scores, bool allow_repeats=True):
        """
        Performs a Viterbi search over the scores for each tag using
        the transitions matrix. If a matrix wasn't supplied, 
        it will return the tags with the highest scores individually.
        """
        if self.transitions is None:
            best_scores = scores.argmax(1)
            
            if allow_repeats:
                return best_scores
            
            # we must find the combination of tags that maximizes the probabilities
            logadd = np.log(np.sum(np.exp(scores), 1))
            logprobs = (scores.T - logadd).T
            counts = np.bincount(best_scores)
            
            while counts.max() != 1:
                # find the tag with the most conflicting args
                conflicting_tag = counts.argmax()
                
                # arguments with that tag as current maximum
                args = np.where(best_scores == conflicting_tag)[0]
                
                # get the logprobs for those args having this tag
                conflicting_probs = logprobs[args, conflicting_tag]
                
                # find the argument with the highest probability for that tag
                highest_prob_arg = args[conflicting_probs.argmax()] 
                
                # set the score for other arguments in that tag to a low value
                other_args = args[args != highest_prob_arg]
                scores[other_args, conflicting_tag] = -1000
                
                # and find the new maxes, without recalculating probabilities
                best_scores = scores.argmax(1)
                counts = np.bincount(best_scores)
            
            return best_scores
        
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
    
    def _create_target_lookup(self):
        """
        Creates a lookup table with the window value for each different distance
        to the target token (target_dist_lookup) and one with the precomputed
        values in the convolution layer (target_convolution_lookup). 
        """
        # consider padding. if the table has 10 entries, with a word window of 3,
        # we would have to consider up to the distance of 11, because of the padding.
        num_distances = self.target_dist_table.shape[0] + self.word_window_size - 1
        self.target_dist_lookup = np.empty((num_distances, 
                                            self.word_window_size * self.target_dist_table.shape[1]))
        self.target_dist_offset = num_distances / 2
        window_from = 0
        window_to = self.target_dist_table.shape[1] 
        for i in range(self.word_window_size):
            # each token in the window will is shifted in relation to the middle one
            shift = i - self.half_window
            
            # discount half window size because of the extra distances we added for padding
            inds = np.arange(shift, num_distances + shift) - self.half_window
            inds = np.clip(inds, 0, self.target_dist_table.shape[0] - 1)
            self.target_dist_lookup[:,window_from : window_to] = self.target_dist_table[inds,]
            
            window_from = window_to
            window_to += self.target_dist_table.shape[1]
        
        self.target_convolution_lookup = self.target_dist_lookup.dot(self.target_dist_weights)
        
    
    def _create_pred_lookup(self):
        """
        Creates a lookup table with the window value for each different distance
        to the predicate token (pred_dist_lookup) and one with the precomputed
        values in the convolution layer (pred_convolution_lookup). 
        """
        # consider padding. if the table has 10 entries, with a word window of 3,
        # we would have to consider up to the distance of 11, because of the padding.
        num_distances = self.pred_dist_table.shape[0] + self.word_window_size - 1
        self.pred_dist_lookup = np.empty((num_distances, 
                                          self.word_window_size * self.pred_dist_table.shape[1]))
        self.pred_dist_offset = num_distances / 2
        window_from = 0
        window_to = self.pred_dist_table.shape[1] 
        for i in range(self.word_window_size):
            # each token in the window will is shifted in relation to the middle one
            shift = i - self.half_window
            
            # discount half window size because of the extra distances we added for padding
            inds = np.arange(shift, num_distances + shift) - self.half_window
            inds = np.clip(inds, 0, self.pred_dist_table.shape[0] - 1)
            self.pred_dist_lookup[:,window_from : window_to] = self.pred_dist_table[inds,]
            
            window_from = window_to
            window_to += self.pred_dist_table.shape[1] 
        
        self.pred_convolution_lookup = self.pred_dist_lookup.dot(self.pred_dist_weights)
    
    def _create_convolution_lookup(self, sentence, training):
        """
        Creates a lookup table storing the values found by each
        convolutional neuron before summing distance features.
        The table has the format len(sent) x len(convol layer)
        Biases are not included.
        """
        cdef np.ndarray padded_sentence
        
        # add padding to the sentence
        if self.word_window_size > 1:
            padded_sentence = np.vstack((self.pre_padding,
                                         sentence,
                                         self.pos_padding))
        else:
            padded_sentence = sentence
        
        self.convolution_lookup = np.empty((len(sentence), self.hidden_size))
        
        # first window
        cdef np.ndarray window = padded_sentence[:self.word_window_size]
        cdef np.ndarray input_data
        input_data = np.concatenate(
                                    [table[index] 
                                     for token_indices in window
                                     for index, table in zip(token_indices, 
                                                             self.feature_tables)
                                     ]
                                    )
        
        self.convolution_lookup[0] = self.hidden_weights.dot(input_data)
        if training:
            # store the values of each input -- needed when adjusting features
            self.input_sent_values[0] = input_data
        
        cdef np.ndarray[FLOAT_t, ndim=1] new_data
        for i, element in enumerate(padded_sentence[self.word_window_size:], 1):
            new_data = np.concatenate([table[index] for 
                                       index, table in zip(element, self.feature_tables)])
            
            # slide the window to the next element
            input_data = np.concatenate((input_data[self.features_per_token:], 
                                         new_data))
            
            self.convolution_lookup[i] = self.hidden_weights.dot(input_data)
            if training:
                self.input_sent_values[i] = input_data
        
