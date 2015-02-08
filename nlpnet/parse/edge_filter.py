# -*- coding: utf-8 -*-

import itertools
import logging
import cPickle
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import OneHotEncoder


class EdgeFilter(object):
    """
    Class for filtering out unlikely dependency edges. It encapsulates a fast linear
    classifier to be used as a pre-processing step.
    
    The classifier deals with data instances built as:
    [modifier head distance]
    
    where modifier and head are both represented as the concatenation of a dense
    embedding vector (for the word itself) and sparse vectors representing discrete
    attributes (usually, POS).  
    distance is represented as a sparse vector. 
    """
    def __init__(self, feature_table, max_dist, window_size=3, filename=None):
        """
        Constructor. Unlike other classifiers in nlpnet, EdgeFilter only 
        uses embeddings to represent words. Other attributes are 
        represented as sparse vectors. This is because EdgeFilter doesn't 
        use neural networks.
        
        :param feature_table: feature with word embeddings
        :param max_dist: the maximum distance treated as an independent feature
            (e.g., if 4, there will be a feature for each distance 0-4 and another 
            for 5+, both positive and negative)
        :param window_size: the size of the window containing each head and modifier
            candidate
        :param filename: file to save the model on
        """
        self.feature_table = feature_table
        self.max_dist = max_dist
        self.window_size = 3
        self.filename = filename
        
        # when determining the representation of the distance between two tokens, we do:
        # raw_diff = position_token_1 - position_token_2
        # raw_diff is then clipped to fit inside the maximum distance and added distance_offset,
        # so that the largest negative distance is mapped to 0, which can be interpreted
        # as an index to an array 
        self.num_sparse_dist_features = 2 * max_dist + 3
        self.distance_offset = max_dist + 1
    
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
        shape = (len(sentence), self.features_per_token)
        lookup = np.empty(shape)
        index = self.feature_table.shape[1]
        
        # dense vectors for words
        lookup[:, :index] = self.feature_table.take(sentence[:, 0], axis=0)
        # sparse vectors for other attributes (not worth to use scipy sparse representation)
        lookup[:, index:] = self.encoder.transform(sentence[:, 1:]).todense()
            
        return lookup
    
    def _create_instances(self, sentence, heads=None):
        """
        Create all the training instances from a sentence, yielding
        the vectors used by the classifier.
        
        :param heads: if given, returns a tuple with the training instances
            and the classes (True and False)
        """
        num_instances = len(sentence) ** 2
        instances = np.zeros((num_instances, self.instance_size))
        training = heads is not None
        if training:
            classes = np.zeros(num_instances, dtype=np.int8)
        
        # lookup for the full representation of each token
        sentence_lookup = self._create_sentence_lookup(sentence)
        
        ind_from_row = 0
        
        # we'll take each pair (i, j) from the sentence
        # (i, j) != (j, i)
        for i in range(len(sentence)):
            # each instance is represented as 
            # [modifier_features head_features distance_features]
            
            # the features for token i will appear len(sent) times
            # each time together with a different j
            ind_to_row = ind_from_row + len(sentence)
            instances[ind_from_row:ind_to_row, 0:self.features_per_token] = sentence_lookup[i]
            
            ind_to_column = 2 * self.features_per_token
            instances[ind_from_row:ind_to_row, self.features_per_token:ind_to_column] = sentence_lookup
            ind_from_column = ind_to_column
            
            # distances from each token to i
            distances = np.arange(len(sentence)) - i
            np.clip(distances, -self.distance_offset, self.distance_offset, distances)
            distances += self.distance_offset
            
            distance_submatrix = instances[ind_from_row:ind_to_row, ind_from_column:]
            distance_submatrix[np.arange(len(sentence)), distances] = 1.0
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
    
    @property
    def threshold(self):
        """
        The threshold probability used by the filter.
        """
        return self._threshold
    
    @threshold.setter
    def threshold(self, value):
        self._threshold = value
        
        # the probability is determined from the score using the logistic function:
        # p = 1 / (1 + exp(-score))
        # the threshold score can be determined as:
        # threshold_score = log(p) - log(1 - p)
        self._threshold_score = np.log(value) - np.log(1 - value)
    
    def filter(self, sentence):
        """
        Apply the learned filter to filter out dependency edges in the sentence.
        
        :param sentence: a numpy array of tokens
        :return: a 2-dim array where each cell (i, j) has False if the classifier is 
            confident that it isn't a valid edge with the given error threshold. 
            Possible edges have value True. (i, j) means token i has head j.
        """
        sentence_size = len(sentence)
        instances = self._create_instances(sentence)
        
        scores = self.classifier.decision_function(instances)
        scores = scores.reshape((sentence_size, sentence_size))
        return scores > self._threshold_score
    
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
        
        self.threshold = threshold
        
        for sent, sent_heads in zip(sentences, heads):
            # False means classifier is confident that there is no edge there
            # we want no actual edge marked as False, but a few non-edges as True
            # are no problem
            answers = self.filter(sent)
            
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
        if filename is None:
            filename = self.filename
        
        with open(filename, 'wb') as f:
            # self.__dict__ contains every attribute in self
            cPickle.dump(self.__dict__, f, 2)
    
    @classmethod
    def load(cls, filename):
        """
        Load a model from the given filename.
        """
        with open(filename, 'rb') as f:
            data = cPickle.load(f)
        
        # data is the pickled __dict__ attribute 
        max_dist = data['max_dist']
        feature_table = data['feature_table']
        edge_filter = cls(feature_table, max_dist, filename)
        
        # just update it, except for "filename", which may be different
        del data['filename']
        edge_filter.__dict__.update(data)
        
        logger = logging.getLogger('Logger')
        logger.info('Loaded dependency filter')
        return edge_filter
    
    def _fit_encoder(self, sentences):
        """
        Fit the OneHotEncoder used with discrete attributes.
        
        :param sentences: matrix containing a token in each row.
        """
        # use a one-hot encoder for attributes after the word indices 
        # "sparse" argument in enconder constructor is available in sklearn 0.15
        attributes_values = np.concatenate([sent[:, 1:] for sent in sentences])
        self.encoder = OneHotEncoder()
        self.encoder.fit(attributes_values)
        
        # get how many extra attributes are there (besides the word index to the feature table)
        num_sparse_features = self.encoder.feature_indices_[-1]
        self.features_per_token = self.feature_table.shape[1] + num_sparse_features
        self.instance_size = 2 * (self.features_per_token) + self.num_sparse_dist_features
    
    def train(self, sentences, heads, loss_function):
        """
        Train the model to detect unlikely edges.
        """
        self._fit_encoder(sentences)
        self.classifier = SGDClassifier(loss_function, class_weight='auto')
        
        batch_size = self._find_batch_size(sentences)
        num_batches = int(np.ceil(len(sentences) / batch_size))
        
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
            
