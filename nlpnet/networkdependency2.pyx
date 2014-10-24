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
    
    # lookup for all embeddings in a window
    cdef readonly np.ndarray window_lookup
    
    # distance
    cdef readonly np.ndarray distance_lookup, distance_table
    cdef int distance_offset
    
    # arrays storing the different inputs during training
    cdef input 
    
    # the scores of all possible dependency edges among tokens
    cdef readonly np.ndarray dependency_scores
    
    def __init__(self, feature_tables, word_window, hidden_size, 
                 num_distance_features, max_distance):
        """
        Constructor
        """
        self.word_window_size = word_window
        self.hidden_size = hidden_size
        self.output_size = 1
        self.feature_tables = feature_tables
        self.features_per_token = sum(table.shape[1] for table in feature_tables)
        
        num_distance_entries = (2 * max_distance) + 3
        self.distance_offset = num_distance_entries / 2
        self.distance_table = (0.2) * np.random.random((num_distance_entries, 
                                                        num_distance_features)) - 0.1
        
        features_per_window = self.features_per_token * word_window
        self.input_size = 2 * features_per_window + num_distance_features
        
        self.learning_rate = 0
        self.learning_rate_features = 0
        
        self.transitions = None
        self.hidden_weights = None
        
        high = 2.38 / np.sqrt(self.input_size) # [Bottou-88]
        self.modifier_weights = np.random.uniform(-high, high, (features_per_window, hidden_size))
        self.head_weights = np.random.uniform(-high, high, (features_per_window, hidden_size))
        self.distance_weights = np.random.uniform(-high, high, (num_distance_features, hidden_size))
        self.hidden_bias = np.random.uniform(-high, high, hidden_size)
        
        high = 2.38 / np.sqrt(hidden_size)
        self.output_weights = np.random.uniform(-high, high, hidden_size)
        self.output_bias = np.random.uniform(-high, high)
        
        self.validation_sentences = None
        self.validation_tags = None
        
        self.use_learning_rate_decay = False
    
    def tag_sentence(self, np.ndarray sentence, np.ndarray heads=None):
        """
        Run the network for the unlabeled dependency task.
        A graph with all weights for possible dependencies is built 
        and the final answer is obtained applying the Chu-Liu-Edmond's
        algorithm.
        """        
        training = heads is not None
        
        self._create_sentence_lookup(sentence)
        
        num_tokens = len(sentence)
        # dependency_weights [i, j] has the score for token i having j as a head.
        # the main diagonal has the values for dependencies from the root and is 
        # later copied to the last column for easier processing
        self.dependency_scores = np.empty((num_tokens, num_tokens + 1))
        
        # compute values in the hidden layer for all heads and modifiers
        head_values = self.window_lookup.dot(self.head_weights)
        modifier_values = self.window_lookup.dot(self.modifier_weights)
        
        if training:
            # matrices to store the value computed by each hidden neuron for each
            # head-modifier pair
            self.layer2_sent_values = np.empty((len(sentence), self.hidden_size))
            self.hidden_sent_values = np.empty((len(sentence), self.hidden_size))
        
        # calculate the score for each (head, modifier) pair
        for modifier in range(num_tokens):
            
            # run all head candidates at once (faster)
            dist_indices = np.arange(len(sentence)) - modifier
            dist_values = self.distance_lookup.take(dist_indices + self.distance_offset,
                                                    0, mode='clip')
            
            # the hidden layer will output n results, with n = len(sentence)
            hidden_sum = head_values + modifier_values[modifier] + dist_values + self.hidden_bias
            self.hidden_values = hardtanh(hidden_sum, inplace=not training)
            if training:
                # layer2_sent_values stores the values before non-linearity
                self.layer2_sent_values[modifier] = hidden_sum
                self.hidden_sent_values[modifier] = self.hidden_values
            
            output = self.hidden_values.dot(self.output_weights) + self.output_bias
            self.dependency_scores[modifier, :-1] = output
            
            if training:
                head = heads[modifier]
                if self._calculate_gradients(head, output):
                    self._backpropagate(modifier)
                    self._calculate_input_deltas(sentence, token, filtered=candidates)
                    self._adjust_weights(token, filtered=candidates)
                    self._adjust_features(sentence, token)
        
        # copy dependency weights from the root to each token to the last column and
        # effectively ignore the main diagonal (dependency to the token itself)
        self.dependency_scores[np.arange(num_tokens), 
                               -1] = self.dependency_scores.diagonal()
        np.fill_diagonal(self.dependency_scores, -np.Infinity)
        answer = self._find_maximum_spanning_tree()
        
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
        
        # backpropagation expects a 2-dim matrix
        new_shape = (self.net_gradients.shape[0], 1)
        self.net_gradients = self.net_gradients.reshape(new_shape)
        
        return True
    
    def _backpropagate(self, sentence, modifier):
        """
        Backpropagate the gradients of the cost.
        
        :param modifier: the index of the modifier in the sentence. It is assumed
            that adjustments are done w.r.t. all head candidates.
        """
        output_deltas = self.net_gradients.dot(self.hidden_sent_values)
        output_bias_delta = self.net_gradients.sum()
        self.output_weights += output_deltas * self.learning_rate
        self.output_bias += output_bias * self.learning_rate
        
        # this is the same as np.outer(x, y), but faster
        partial_gradient = self.net_gradients.reshape((len(self.net_gradients),1))
                            .dot(self.output_weights.reshape((1, self.hidden_size)))
        # hidden_gradients -> (len, hidden_size)
        hidden_gradients = hardtanhd(self.layer2_sent_values) * partial_gradient
        
        # the whole window lookup table was the input to the hidden layer
        # deltas -> (head_window_size, hidden_size)
        head_weight_deltas = self.window_lookup.T.dot(hidden_gradients)
        modifier_single_input = self.window_lookup[modifier]
        modifier_input = modifier_single_input.reshape(len(modifier_single_input), 1)
        modifier_weights_deltas = modifier_input * hidden_gradients.sum(0)
        
        dist_indices = np.arange(len(self.net_gradients)) - modifier
        dist_input = self.distance_table.take(dist_indices + self.distance_offset,
                                              0, mode='clip').T
        distance_weights_deltas = dist_input.dot(hidden_gradients)
        hidden_bias_deltas = hidden_gradients.sum(0)
        
        head_gradients = hidden_gradients.dot(self.head_weights.T) * self.learning_rate_features
        modifier_gradients = hidden_gradients.dot(self.modifier_weights.T) * self.learning_rate_features
        distance_gradients = hidden_gradients.dot(self.distance_weights.T) * self.learning_rate_features
        
        self.head_weights += head_weight_deltas * self.learning_rate
        self.modifier_weights += modifier_weights_deltas * self.learning_rate
        self.distance_weights += distance_weights_deltas * self.learning_rate
        
        cdef np.ndarray padded_sentence = np.concatenate((self.pre_padding,
                                                          sentence,
                                                          self.pos_padding))
        
        # modifier gradients refers to adjustments in the same tokens for every line
        modifier_gradients_sum = modifier_gradients.sum(0)
        modifier_gradients_sum = modifier_gradients_sum.reshape(self.word_window_size,
                                                                self.features_per_token)
        
        # aggregate all window positions for head deltas
        padded_sentence_deltas = np.zeros((len(padded_sentence), self.features_per_token))
        from_feature = 0
        for i in range(self.word_window_size):
            until_token = i + len(sentence)
            until_feature = from_feature + self.features_per_token
            padded_sentence_deltas[i:until_token] += head_gradients[:, from_feature:until_feature]
        
        # combine modifier and head deltas
        half_window = self.word_window_size / 2
        padded_sentence_deltas[modifier - half_window:modifier + half_window + 1] += modifier_gradients_sum
        
        # apply deltas to feature tables
        from_feature = 0
        for i, token in padded_sentence:
            for j, table in enumerate(self.feature_tables):
                until_feature = from_feature + table.shape[1]
                index = token[j]
                table[j] += padded_sentence_deltas[i, from_feature:until_feature]
                from_feature = until_feature
        
        # distance features adjustments
        dists = np.arange(len(sentence)) - modifier
        clipped_dists = np.clip(dists, 0, len(self.distance_table)) + self.distance_offset
        # distance_gradients is (num_tokens, num_distance_features)
        
                
            
    
    def _create_distance_lookup(self):
        """
        Creates a lookup table containing the values computed in the hidden
        layer for each distance.
        """
        self.distance_lookup = self.distance_table.dot(self.distance_weights)
    
    def _create_sentence_lookup(self, sentence):
        """
        Creates a sentence lookup table, such that the row i has the input features
        for the window centered at the token i of the sentence.
        """
        # the Network class creates a lookup table that has each token without its window
        super(DependencyNetwork2, self)._create_sentence_lookup(sentence)
        
        window_lookup_size = self.word_window_size * self.features_per_token
        self.window_lookup = np.empty((len(sentence), window_lookup_size))
        
        from_column = 0
        for i in range(self.word_window_size):
            until_column = (i + 1) * self.features_per_token
            until_token = i + len(sentence)
            self.window_lookup[:, from_column:until_column] = self.sentence_lookup[i:until_token]
            
            from_column = until_column
    
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
        
        num_vertices = self.dependency_scores.shape[1]
        outside = np.array([x for x in range(num_vertices) if x not in cycle])
        cycle = np.array(list(cycle))
        
        cdef np.ndarray[FLOAT_t, ndim=2] outgoing_weights
        
        # adjustments will be made on incoming and outgoing edges
        # pick the part of the weight matrix that contain them
        
        # if len(outside) == 1, it means all vertices except for the loop are in a cycle
        if len(outside) > 1:
            # weird index array we need in order to properly use fancy indexing
            # -1 because we can't take the root now
            outside_inds = np.array([[i] for i in outside[:-1]])
            outgoing_weights = self.dependency_scores[outside_inds, cycle]
            
            # the cycle should have only one outgoing edge for each vertex outside it
            # so, search the maximum outgoing edge of each outside vertex
            max_outgoing_inds = outgoing_weights.argmax(1)
            max_outgoing_weights = outgoing_weights.max(1)
            
            # set every outgoing weight to -inf and then restore the highest ones
            outgoing_weights[:] = -np.Infinity
            outgoing_weights[np.arange(len(outside_inds)), 
                             max_outgoing_inds] = max_outgoing_weights
            self.dependency_scores[outside_inds, cycle] = outgoing_weights
        
        # now, adjust incoming edges. the incoming edge from each vertex v 
        # (outside the cycle) to v' (inside) is reweighted as: 
        # s(v, v') = s(v, v') - s(head(v'), v') + s(c)
        # and then we pick the highest edge for each outside vertex
        # s(c) = sum_v s(head(v), v)
        cycle_inds = np.array([[i] for i in cycle])
        cdef np.ndarray[FLOAT_t, ndim=2] incoming_weights
        incoming_weights = self.dependency_scores[cycle_inds, outside]
        
        cycle_score = 0
        for i, vertex in enumerate(cycle):
            head_to_v = self.dependency_scores[vertex, heads[vertex]]
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
        self.dependency_scores[vertex_leaving_cycle, old_head] = -np.Infinity
        
        # analagous to the outgoing weights
        incoming_weights[:] = -np.Infinity
        incoming_weights[max_incoming_inds,
                         np.arange(len(outside))] = max_incoming_weights
        self.dependency_scores[cycle_inds, outside] = incoming_weights
    
    def _find_maximum_spanning_tree(self):
        """
        Run the Chu-Liu / Edmond's algorithm in order to find the highest
        scoring dependency tree from the given dependency graph weight.
        
        :param weights: a 2-dim matrix containing at 
        :returns: a 1-dim array with the head of each token in the sentence
        """
        # pick the highest dependency for each word
        heads = self.dependency_scores.argmax(1)
        
        # check if there are cycles. if there isn't any, we're done
        cycles = self._find_cycles(heads)
        
        for cycle in cycles:
            # resolve each cycle c
            self._contract_cycle(heads, cycle)
        
        return heads
