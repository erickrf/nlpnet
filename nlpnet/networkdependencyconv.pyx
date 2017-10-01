# -*- coding: utf-8 -*-

"""
A convolutional neural network for NLP tagging tasks such as dependency
parsing, where each token has another (or root) as a head. 
"""

import numpy as np
cimport numpy as np

cdef class ConvolutionalDependencyNetwork(ConvolutionalNetwork):
    
    # the weights of all possible dependency among tokens
    cdef readonly np.ndarray dependency_weights
    
    # validation data
    cdef validation_heads
    
    def save(self):
        """
        Saves the neural network to a file.
        It will save the weights, biases, sizes, padding and 
        distance tables, and other feature tables.
        """
        data = self._generate_save_dict()
        
        np.savez(self.network_filename, **data)
    
    @classmethod
    def load_from_file(cls, filename):
        """
        Loads the neural network from a file.
        It will load weights, biases, sizes, padding and 
        distance tables, and other feature tables.
        """
        data = np.load(filename, encoding='bytes')
        nn = cls._load_from_file(data, filename)
        
        return nn
    
    def train(self, list sentences, list heads, int epochs, 
              int epochs_between_reports=0, float desired_accuracy=0,
              list labels=None):
        """
        Trains the convolutional network. Refer to the basic Network
        train method for detailed explanation. 
        """
        # the ConvolutionalNetwork class was written primarily for SRL
        # every token acts as a predicate, and we don't need to tell it explicitely
        predicates = [np.arange(len(sentence)) for sentence in sentences]
        
        # the last argument in ConvolutionalNetwork.train is actually the argument
        # groups list. We use "labels" here just to signal that there is non-None
        # argument, which is correctly handled by the DependencyNetwork._tag(...) method.
        
        if self.validation_sentences is None:
            self.set_validation_data(sentences, heads, labels)
        
        super(ConvolutionalDependencyNetwork, self).train(sentences, predicates, 
                                                          heads, epochs, 
                                                          epochs_between_reports, 
                                                          desired_accuracy,
                                                          labels)
        
    def set_validation_data(self, list sentences, list heads, list labels=None):
        """
        Sets the data to be used in validation during training. If this function
        is not called before training, the training data itself is used to 
        measure the model's performance.
        
        :param labels: only used when learning labels
        """
        self.validation_sentences = sentences
        self.validation_tags = labels
        self.validation_heads = heads
    
    def _tag_sentence(self, sentence, predicates=None, heads=None, labels=None):
        """
        This function is just an interface to the _tag_sentence signature
        defined in ConvolutionalNetwork.
        """
        if labels is None:
            self._tag_sentence_unlabeled_dependency(sentence, heads)
        else:
            self._tag_sentence_labeled_dependency(sentence, heads, labels)
    
    def _tag_sentence_unlabeled_dependency(self, np.ndarray sentence, np.ndarray heads=None):
        """
        Run the network for the unlabeled dependency task.
        A graph with all weights for possible dependencies is built 
        and the final answer is obtained applying the Chu-Liu-Edmond's
        algorithm.
        """
        training = heads is not None
        self._pre_tagging_setup(sentence, training)
        
        num_tokens = len(sentence)
        # dependency_weights [i, j] has the score for token i having j as a head.
        # the main diagonal has the values for dependencies from the root and is 
        # later copied to the last column for easier processing
        self.dependency_weights = np.empty((num_tokens, num_tokens + 1))
        
        cdef np.ndarray[FLOAT_t, ndim=1] token_scores
        
        # in the SRL parlance, each token is treated as a predicate, because all
        # sentence tokens are scored with respect to it (in order to determine the
        # dependency weights)
        for token in range(num_tokens):
            
            # _sentence_convolution returns a 2-dim array. in dep parsing, 
            # we only have one dimension, so reshape it
            token_scores = self._sentence_convolution(sentence, token, 
                                                      training=training).reshape(num_tokens)
            self.dependency_weights[token, :-1] = token_scores
            
            if training:
                head = heads[token]
                 
                if self._calculate_gradients(head, token_scores):
                    self._backpropagate()
                    self._calculate_input_deltas(sentence, token)
                    self._adjust_weights(token)
                    self._adjust_features(sentence, token)
        
        # copy dependency weights from the root to each token to the last column and
        # effectively ignore the main diagonal (dependency to the token itself)
        self.dependency_weights[np.arange(num_tokens), 
                                -1] = self.dependency_weights.diagonal()
        np.fill_diagonal(self.dependency_weights, -np.Infinity)
        answer = self._find_maximum_spanning_tree()
        
        return answer
    
    def _tag_sentence_labeled_dependency(self, np.ndarray sentence, np.ndarray heads,
                                         np.ndarray labels=None):
        """
        Run the network for labeling pre determined dependency edges between tokens.
        This is similar to the classification step in SRL.
        """
        cdef np.ndarray[FLOAT_t, ndim=1] answer
        cdef np.ndarray[FLOAT_t, ndim=2] scores
        training = labels is not None
        self._pre_tagging_setup(sentence, training)
        
        answer = np.zeros(len(sentence))
        
        # as in unlabeled dependency, each token is treated as a predicate from the
        # SRL point of view. The only target is its head
        for token in range(len(sentence)):
            head = heads[token]
            
            # weird format just to take advantage of the SRL classification code
            # it means that the target starts at position *head* and ends at *head*
            head = [[head, head]]
            
            # it will return a 2-dim array, but we only have one target
            # argmax() works as expected
            scores = self._sentence_convolution(sentence, token, head, training)
            answer[token] = scores.argmax()
            
            if training:
                label = labels[token]
                if self._calculate_gradients_classify([label], scores):
                    self._backpropagate()
                    self._calculate_input_deltas(sentence, token, head)
                    self._adjust_weights(token, head)
                    self._adjust_features(sentence, token)
        
        
        return answer
    
    def _calculate_gradients(self, gold_head, scores):
        """
        Calculate the gradients to be applied in the backpropagation. Gradients
        are calculated after the network has output the scores for assigning 
        each token as head of a given token. 
        
        We aim at maximizing the log probability of the right head:
        log(p(head)) = score(head) - logadd(scores for all heads)
        
        :param gold_head: the index of the token that should have the highest 
            score
        :param scores: the scores output by the network
        :returns: if True, normal gradient calculation was performed.
            If False, the error was too low and weight correction should be
            skipped. 
        """
        # first, set the gradient at each token to 
        # -exp(score(token)) / sum_j exp(score(token_j))
        # i.e., the negative of its probability
        cdef np.ndarray[FLOAT_t, ndim=1] exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores)
        self.net_gradients = -exp_scores / exp_sum
        error = 1 + self.net_gradients[gold_head]
        self.error += error
        
        # check if the error is too small - if so, not worth to continue
        if error <= 0.01:
            self.skips += 1
            return False
        
        # and add 1 to the right head
        self.net_gradients[gold_head] += 1
        
        # the ConvolutionalNetwork class deals with multi dimensional gradients
        # (because of more than one output neuron), so let's reshape
        new_shape = (self.net_gradients.shape[0], 1)
        self.net_gradients = self.net_gradients.reshape(new_shape)  
        
        return True
    
    def _validate(self):
        """
        Evaluate the network performance by token hit and whole sentence hit.
        """
        hits = 0
        num_tokens = 0
        sentence_hits = 0
        
        for i in range(len(self.validation_sentences)):
            sent = self.validation_sentences[i]
            heads = self.validation_heads[i]
            sentence_hit = True
            
            if self.validation_tags is None:
                # unlabeled dependency
                answer = self._tag_sentence_unlabeled_dependency(sent)
                gold_tags = heads
            else:
                # labeled dependency
                gold_tags = self.validation_tags[i]
                answer = self._tag_sentence_labeled_dependency(sent, heads)
                
            for j in range(len(gold_tags)):
                net_tag = answer[j]
                gold_tag = gold_tags[j]
                
                if net_tag == gold_tag or (gold_tag == j and net_tag == len(sent)):
                    hits += 1
                else:
                    sentence_hit = False
            
            if sentence_hit:
                sentence_hits += 1
            num_tokens += len(sent)
        
        self.accuracy = float(hits) / num_tokens
        self.sentence_accuracy = float(sentence_hits) / len(self.validation_sentences)
        
    
    def _average_error(self):
        """
        Average the network error over tokens.
        """
        self.error = self.error / self.num_tokens
    
    def _print_epoch_report(self, int num):
        """
        Reports the status of the network in the given training
        epoch, including error, token and sentence accuracy.
        """
        logger = logging.getLogger("Logger")
        logger.info("%d epochs   Error: %f   Token accuracy: %f   " \
                    "Sentence accuracy: %f    " \
                    "%d corrections skipped   " \
                    "Learning rate: %f"  % (num,
                                            self.error,
                                            self.accuracy,
                                            self.sentence_accuracy,
                                            self.skips,
                                            self.learning_rate))
    
    def tag_sentence(self, np.ndarray sentence, np.ndarray heads=None):
        """
        If heads is not given, compute the dependency edges in the sentence.
        If it is given, compute the label of each dependency.
        
        :returns: a numpy 1-dim array with the head for each token or label
            of each edge. In the first case, a dependency from the root is 
            represented as a value equal to the sentence length. 
        """
        if heads is None:
            return self._tag_sentence_unlabeled_dependency(sentence)
        else:
            return self._tag_sentence_labeled_dependency(sentence, heads)
    
    def _find_cycles(self, np.ndarray graph):
        """
        Check if the given graph has cycles and returns the first one
        to be found.
        
        :param graph: an array where graph[i] has the number of a
            vertex with an incoming connection to i
        """
        # this set stores all vertices with a valid path to the root
        reachable_vertices = set()
        
        # vertices known to be unreachable from the root, i.e., in a cycle
        vertices_in_cycles = set()
        
        # vertices currently being evaluated, not known if they're reachable
        visited = set()
        
        cycles = []
        
        # the directions of the edges don't matter if we only want to find cycles
        for vertex in range(len(graph)):
            if vertex in reachable_vertices or vertex in vertices_in_cycles:
                continue
            
            cycle = self._find_cycle_recursive(graph, vertex, visited, 
                                               reachable_vertices, vertices_in_cycles)
            if cycle is not None:
                cycles.append(cycle)
        
        return cycles
            
    def _find_cycle_recursive(self, np.ndarray graph, int vertex, set visited, 
                              set reachable, set unreachable):
        """
        Auxiliary recursive function for searching the graph for cycles.
        It returns the first cycle it can find starting from the given
        vertex, or None.
        """
        next_vertex = graph[vertex]
        root = len(graph)
        visited.add(vertex)
        
        if next_vertex == root or next_vertex in reachable:
            # vertex linked to root
            reachable.update(visited)
            visited.clear()
            cycle = None
        
        elif next_vertex in visited:
            # cycle detected! return all vertices in it
            visited.clear()
            cycle = set([vertex])
            while next_vertex != vertex:
                cycle.add(next_vertex)
                next_vertex = graph[next_vertex]
            
            unreachable.update(cycle)
        
        elif next_vertex in unreachable:
            # vertex linked to an existing cycle, but not part of it
            # (if it were, it should have been filtered out in _find_cycles)
            visited.clear()
            cycle = None
        
        else:
            # continue checking
            cycle = self._find_cycle_recursive(graph, next_vertex, visited, 
                                               reachable, unreachable)
        
        return cycle
    
    def _contract_cycle(self, heads, cycle):
        """
        Contract the given cycle in the dependency graph.
        
        :param heads: list of the heads of each token, such that
            heads[i] contains the head for the i-th token
        :param cycle: a set containing the numbers of the vertices
            in the cycle
        """
        # each cell i, j has the score for token i having j as its head.
        
        num_vertices = self.dependency_weights.shape[1]
        outside = np.array([x for x in range(num_vertices) if x not in cycle])
        cycle = np.array(list(cycle))
        
        cdef np.ndarray[FLOAT_t, ndim=2] outgoing_weights
        
        # adjustments will be made on incoming and outgoing edges
        # pick the part of the weight matrix that contain them
        
        # if len(outside) == 1, it means all vertices except for the root are in a cycle
        if len(outside) > 1:
            # weird index array we need in order to properly use fancy indexing
            # -1 because we can't take the root now
            outside_inds = np.array([[i] for i in outside[:-1]])
            outgoing_weights = self.dependency_weights[outside_inds, cycle]
            
            # the cycle should have only one outgoing edge for each vertex outside it
            # so, search the maximum outgoing edge to each outside vertex
            max_outgoing_inds = outgoing_weights.argmax(1)
            max_outgoing_weights = outgoing_weights.max(1)
            
            # set every outgoing weight to -inf and then restore the highest ones
            outgoing_weights[:] = -np.Infinity
            outgoing_weights[np.arange(len(outside_inds)), 
                             max_outgoing_inds] = max_outgoing_weights
            self.dependency_weights[outside_inds, cycle] = outgoing_weights
        
        # now, adjust incoming edges. the incoming edge from each vertex v 
        # (outside the cycle) to v' (inside) is reweighted as: 
        # s(v, v') = s(v, v') - s(head(v'), v') + s(c)
        # and then we pick the highest edge for each outside vertex
        # s(c) = sum_v s(head(v), v)
        cycle_inds = np.array([[i] for i in cycle])
        cdef np.ndarray[FLOAT_t, ndim=2] incoming_weights
        incoming_weights = self.dependency_weights[cycle_inds, outside]
        
        cycle_score = 0
        for i, vertex in enumerate(cycle):
            head_to_v = self.dependency_weights[vertex, heads[vertex]]
            cycle_score += head_to_v
            incoming_weights[i] -= head_to_v
        
        max_incoming_inds = incoming_weights.argmax(0)
        max_incoming_weights = incoming_weights.max(0)
        # we leave the + s(c) to the end
        max_incoming_weights += cycle_score
        
        # the vertex with the maximum weighted incoming edge now changes
        # its head, thus breaking the cycle
        new_head_ind = max_incoming_weights.argmax()
        vertex_leaving_cycle_ind = max_incoming_inds[new_head_ind]
        
        new_head = outside[new_head_ind]
        vertex_leaving_cycle = cycle[vertex_leaving_cycle_ind]
        old_head = heads[vertex_leaving_cycle]
        heads[vertex_leaving_cycle] = new_head
        self.dependency_weights[vertex_leaving_cycle, old_head] = -np.Infinity
        
        # analagous to the outgoing weights
        incoming_weights[:] = -np.Infinity
        incoming_weights[max_incoming_inds,
                         np.arange(len(outside))] = max_incoming_weights
        self.dependency_weights[cycle_inds, outside] = incoming_weights
    
    def _find_maximum_spanning_tree(self):
        """
        Run the Chu-Liu / Edmond's algorithm in order to find the highest
        scoring dependency tree from the dependency graph weights.
        
        :returns: a 1-dim array with the head of each token in the sentence
        """
        # pick the highest scoring dependency for each word
        heads = self.dependency_weights.argmax(1)
        
        # check if there are cycles. if there isn't any, we're done
        cycles = self._find_cycles(heads)
        
        for cycle in cycles:
            # resolve each cycle c
            self._contract_cycle(heads, cycle)
        
        return heads
    

        