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
        shape = (len(sentence), np.sum(table.shape[1] for table in self.feature_tables))
        lookup = np.empty(shape)
        ind_from = 0
        
        for i, table in enumerate(self.feature_tables):
            # take the values from each feature table for all tokens at a time
            table_indices = sentence.take(i, axis=1)
            features = table.take(table_indices, 0)
            
            ind_to = ind_from + table.shape[1]
            lookup[:, ind_from:ind_to] = features
            ind_from = ind_to
            
        return lookup
    
    def _create_instances(self, sentence, heads=None):
        """
        Create all the training instances from a sentence, yielding
        the vectors used by the classifier.
        
        :param heads: if given, returns a tuple with the training instances
            and the classes (True and False)
        """
        num_instances = len(sentence) ** 2
        instances = np.empty((num_instances, self.instance_size))
        training = heads is not None
        if training:
            classes = np.zeros(num_instances)
        
        # lookup for the full representation of each token
        sentence_lookup = self._create_sentence_lookup(sentence)
        
        ind_from_row = 0
        features_per_token = np.sum(table.shape[1] for table in self.feature_tables)
        
        # we'll take each pair (i, j) from the sentence
        # (i, j) != (j, i)
        for i in range(len(sentence)):
            # each instance is represented as 
            # [modifier_features head_features distance_features]
            
            # the features for token i will appear len(sent) times
            # each time together with a different j
            ind_to_row = ind_from_row + len(sentence)
            instances[ind_from_row:ind_to_row, 0:features_per_token] = sentence_lookup[i]
            
            ind_to_column = 2 * features_per_token
            instances[ind_from_row:ind_to_row, features_per_token:ind_to_column] = sentence_lookup
            ind_from_column = ind_to_column
            
            # distances from each token to i
            distances = np.arange(len(sentence)) - i
            dist_features = self.distance_features.take(distances, 0, mode='clip')
            instances[ind_from_row:ind_to_row, ind_from_column:] = dist_features
            ind_from_row = ind_to_row
            
            if training:
                head = heads[i]
                index_head = len(sentence) * i + head
                classes[index_head] = 1
        
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
        :return: a 2-dim array where each cell (i, j) has False if the classifier is 
            confident that it isn't a valid edge with the given error threshold. 
            Possible edges have value True. (i, j) means token i has head j.
        """
        # the probability is determined from the score using the logistic function:
        # p = 1 / (1 + exp(-score))
        # the threshold score can be determined as:
        # threshold_score = log(p) - log(1 - p)
        p = threshold
        threshold_score = np.log(p) - np.log(1 - p)
        sentence_size = len(sentence)
        instances = self._create_instances(sentence)
        
        scores = self.classifier.decision_function(instances)
        scores = scores.reshape((sentence_size, sentence_size))
        return scores > threshold_score
    
    def test(self, sentences, heads, threshold):
        """
        Test the classifier performance on the given data.
        """
        # total number of existing edges
        total_edges = 0
        # total number of edges filtered out by the classifier
        total_filtered = 0
        # number of edges filtered out that shouldn't have been
        wrongly_filtered = 0
        
        for sent, sent_heads in zip(sentences, heads):
            # False means classifier is confident that there is no edge there
            # we want no actual edge marked as False, but a few non-edges as True
            # are no problem
            answers = self.filter(sent, threshold)
            
            # quickly create the gold version of the matrix
            sent_len = len(sent)
            gold = np.zeros((sent_len, sent_len), dtype=np.bool)
            gold[np.arange(sent_len), sent_heads] = True
            
            total_filtered += np.count_nonzero(answers == False)
            mistakes = np.logical_and(np.logical_not(answers), gold)
            wrongly_filtered += np.count_nonzero(mistakes)
            total_edges += gold.size
        
        filtered_percentage = 100 * float(total_filtered) / total_edges
        logger = logging.getLogger('Logger')
        msg = 'Filtered out {:.2f}% of the edges ({:,} out of {:,} edges)'
        logger.info(msg.format(filtered_percentage, total_filtered, total_edges))
        
        # there's one head per token
        total_correct_edges = np.sum(len(sent) for sent in sentences)
        # percentage of sentences an oracle could get right
        oracle = 100.0 * (total_correct_edges - wrongly_filtered) / total_correct_edges
        logger.info('Maximum oracle score (correct edges left): {:.2f}%'.format(oracle))
    
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
        edge_filter = cls(feature_tables, distance_features=distance_table, filename=filename)
        edge_filter.classifier = data['classifier']
        
        return edge_filter
    
    def train(self, sentences, heads, loss_function):
        """
        Train the model to detect unlikely edges.
        """
        batch_size = self._find_batch_size(sentences)
        num_batches = int(np.ceil(len(sentences) / batch_size))
        self.classifier = SGDClassifier(loss_function, class_weight='auto')
        
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
            
