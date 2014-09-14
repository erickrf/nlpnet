# -*- coding: utf-8 -*-

import itertools
import logging
import cPickle
import numpy as np

import nlpnet
from sklearn.linear_model import SGDClassifier
from nlpnet import utils


class EdgeFilter(object):
    def __init__(self, feature_tables, max_dist=None, num_distance_features=None, 
                 distance_features=None, filename=None):
        """
        Constructor. Either provide max_dist and num_distance_features or the
        distance_features table.
        
        :param feature_tables: list of feature tables representing input tokens
        :param max_dist: the maximum distance having an own feature vector
        :param num_distance_features: number of features (vector dimension) 
            for each distance
        :param distance_features: distance feature table.
            if provided, use this feature table for distances
            instead. Usually, only needed in the load method.
        :param filename: file to save the model on
        """
        self.feature_tables = feature_tables
        
        if distance_features is None:
            logger = logging.getLogger("Logger")
            num_vectors = 3 + 2 * max_dist
            logger.info("Generating distance features...")
            distance_features = utils.generate_feature_vectors(num_vectors, num_distance_features)
        else:
            max_dist = (distance_features.shape[0] - 3) / 2 
        self.distance_features = distance_features
        
        self.instance_size = 2 * np.sum(table.shape[1] for table in self.feature_tables)
        self.instance_size += self.distance_features.shape[1]
        
        self.distance_offset = max_dist / 2
        self.filename = filename
    
    def _find_batch_size(self, sentences):
        """
        Determine a good batch size to fit in memory
        """
        space = 10 ** 8
        mean_examples_per_sentence = np.mean([len(sent) ** 2 for sent in sentences])
        
        # 8 bytes is the usual double size
        batch_size = float(space) / (self.instance_size * mean_examples_per_sentence * 8)
        return int(np.floor(batch_size))
    
    def _create_sentence_lookup(self, sentence):
        """
        Create a lookup with the actual feature values for all tokens 
        in a sentence.
        """
        sentence_lookup = np.vstack([np.concatenate([table[index] 
                                                     for index, table in zip(token_indices, self.feature_tables)]) 
                                     for token_indices in sentence])
        
        return sentence_lookup
    
    def _create_instances(self, sentence, heads=None):
        """
        Create all the training instances from a sentence, yielding
        the vectors used by the classifier.
        
        :param heads: if given, returns a tuple with the training instances
            and the classes (True and False)
        """
        num_instances = len(sentence) ** 2
        instances = np.empty((num_instances, self.instance_size))
        instance_counter = 0
        training = heads is not None
        if training:
            classes = np.zeros(num_instances)
        
        # lookup for the full representation of each token
        sentence_lookup = self._create_sentence_lookup(sentence)
        
        for i in range(len(sentence)):
            for j in range(len(sentence)):
                # j: head index
                # i: modifier index
                head_features = sentence_lookup[j]
                modifier_features = sentence_lookup[i]
                
                # distance from head to modifier
                dist = j - i + self.distance_offset
                dist_features = self.distance_features.take(dist,
                                                            0, mode='clip')
                
                instances[instance_counter] = np.concatenate((head_features, 
                                                              modifier_features,
                                                              dist_features))
                if training and heads[i] == j:
                    classes[instance_counter] = 1
                instance_counter += 1
        
        if training:
            return (instances, classes)
        return instances
    
    def _create_training_batch(self, sentences, heads):
        """
        Create a matrix of training examples and their classes.
        """
        num_instances = np.sum(len(sentence) ** 2 for sentence in sentences)
        instances = np.empty((num_instances, self.instance_size))
        classes = np.empty(num_instances)
        ind = 0
        
        for sent, sent_heads in itertools.izip(sentences, heads):
            sent_instances, sent_classes = self._create_instances(sent, sent_heads)
            num_instances = len(sent_instances)
            
            instances[ind:ind + num_instances] = sent_instances
            classes[ind:ind + num_instances] = sent_classes
            ind += num_instances
                        
        return (instances, classes)
    
    def filter(self, sentence, threshold=0.001):
        """
        Apply the learned filter to filter out dependency edges in the sentence.
        
        :param sentence: a numpy array of tokens
        :param threshold: the maximum probability a filtered edge may have
            (i.e., 0.01 means the classifier has 99% certainty that it should
            be filtered out)
        :return: a 2-dim array where each cell (i, j) has True if the classifier is 
            confident that it isn't a valid edge with the given error threshold. 
            Possible edges have value False.
        """
        log_prob = np.log(threshold)
        sentence_size = len(sentence)
        instances = self._create_instances(sentence)
        
        scores = self.classifier.decision_function(instances)
        scores = scores.reshape((sentence_size, sentence_size))
        return scores > log_prob
    
    def save(self, filename=None):
        """
        Save the model to a file. It also saves the feature tables.
        """
        data = {'classifier': self.classifier,
                'feature_tables': self.feature_tables,
                'distance_table': self.distance_features}
        
        if filename is None:
            filename = self.filename
        
        with open(filename, 'wb') as f:
            # not really a network, but we use md.network for consistency
            #TODO: use a more convenient name instead of network
            cPickle.dump(data, f, 2)
    
    @classmethod
    def load(cls, filename):
        """
        Load a model from the given filename.
        """
        with open(filename, 'rb') as f:
            data = cPickle.load(f)
        
        distance_table = data['distance_table']
        feature_tables = data['feature_tables']
        edge_filter = cls(feature_tables, distance_table=distance_table, filename=filename)
        edge_filter.classifier = data['classifier']
        
        return edge_filter
    
    def test(self, sentences, heads):
        """
        Test the classifier performance on the given data.
        """
        pass
        
    
    def train(self, sentences, heads):
        """
        Train the model to detect unlikely edges.
        """
        batch_size = self._find_batch_size(sentences)
        num_batches = int(np.ceil(len(sentences) / batch_size))
        self.classifier = SGDClassifier('log', class_weight='auto')
        
        logger = logging.getLogger('Logger')
        logger.info('Starting training with {} sentences in {} batches'.format(len(sentences), 
                                                                               num_batches))
        
        for batch in range(num_batches):
            logger.info('Batch {}...'.format(batch))
            ind_from = batch * batch_size
            ind_to = (batch + 1) * batch_size
            
            batch_sents = sentences[ind_from:ind_to]
            batch_heads = heads[ind_from:ind_to]
            examples, classes = self._create_training_batch(batch_sents, batch_heads)
            self.classifier.partial_fit(examples, classes, [0, 1])
            
