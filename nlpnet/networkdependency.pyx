# -*- coding: utf-8 -*-

"""
A convolutional neural network for NLP tagging tasks such as dependency
parsing, where each token has another (or itself) as a head. 
"""

import numpy as np
cimport numpy as np

cdef class DependencyNetwork(ConvolutionalNetwork):
    
    # the weights of all possible dependency among tokens
    # public = TEST ONLY!
    cdef public np.ndarray dependency_weights
    
    def __init__(self):
        # test only!
        pass
    
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
        
        num_vertices = self.dependency_weights.shape[0]
        outside = np.array([x for x in range(num_vertices) if x not in cycle])
        cycle = np.array(list(cycle))
        
        # adjustments will be made on incoming and outgoing edges
        # pick the part of the weight matrix that contain them
        
        # weird index array we need in order to properly use fancy indexing
        # -1 because we can't take the root now
        outside_inds = np.array([[i] for i in outside[:-1]])
        cdef np.ndarray[FLOAT_t, ndim=2] outside_weights
        outside_weights = self.dependency_weights[outside_inds, cycle]
        
        # the cycle should have only one outgoing edge for each vertex outside it
        # so, search the maximum outgoing edge of each outside vertex
        max_outgoing_inds = outside_weights.argmax(1)
        max_outgoing_weights = outside_weights.max(1)
        
        # set every outgoing weight to -inf and then restore the highest ones
        outside_weights[:] = -np.Infinity
        outside_weights[np.arange(len(outside_inds)), 
                        max_outgoing_inds] = max_outgoing_weights
        self.dependency_weights[outside_inds, cycle] = outside_weights
        
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
        scoring dependency tree from the given dependency graph weight.
        
        :param weights: a 2-dim matrix containing at 
        :returns: a 1-dim array with the head of each token in the sentence
        """
        # pick the highest dependency for each word
        heads = self.dependency_weights.argmax(1)
        
        # check if there are cycles. if there isn't any, we're done
        cycles = self._find_cycles(heads)
        
        for cycle in cycles:
            # resolve each cycle c
            self._contract_cycle(heads, cycle)
#             heads = self.dependency_weights.argmax(1)
        
        return heads
    
    def _tag_sentence(self, np.ndarray sentence, list tags=None):
        """
        Run the network for the dependency tagging task.
        A graph with all weights for possible dependencies is built 
        and the final answer is obtained applying the Chu-Liu-Edmond's
        algorithm.
        """
        answer = []
        self._pre_tagging_setup(sentence)

        num_tokens = len(sentence)
        # dependency_weights [i, j] has the score for token i having j as a head
        # the main diagonal has the values for edges from the root and is copied to 
        # the last column for easier processing
        self.dependency_weights = np.empty((num_tokens, num_tokens + 1))
        
        cdef np.ndarray[FLOAT_t, ndim=2] token_scores
        
        # in the SRL parlance, each token is treated as a predicate, because all
        # sentence tokens are scored with respect to it (in order to determine the
        # candidate dependency weight)
        for token in range(num_tokens):
            token_scores = self._sentence_convolution(sentence, token)
            
            # _sentence_convolution returns a 2-dim array. in dep parsing, 
            # we only have one dimension, so take the index 0
            self.dependency_weights[token, :-1] = token_scores[0]
        
        self.dependency_weights[np.arange(num_tokens), 
                                -1] = self.dependency_weights.diagonal()
        np.fill_diagonal(self.dependency_weights, -np.Infinity)
    
        