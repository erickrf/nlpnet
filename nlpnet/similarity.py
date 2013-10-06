# -*- coding: utf-8 -*-

"""
Lists similar words.
"""

import cPickle
from argparse import ArgumentParser
import numpy as np

import utils
import config

def cosine_similarity(vec1, vec2):
    """
    Returns the cosine similarity between two vectors.
    """
    dot_prod = vec1.dot(vec2)
    magnitude1 = np.linalg.norm(vec1)
    magnitude2 = np.linalg.norm(vec2)
    
    return dot_prod / (magnitude1 * magnitude2)

def load_word_dict():
    """
    Loads the word dictionary from its default file.
    """
    with open(config.FILES['word_dict_dat'], 'rb') as f:
        word_dict = cPickle.load(f)
    
    return word_dict


def find_similar_words(feature_table, index, num=5):
    """
    Finds the most similar words to the one indicated. 
    :param feature_table: 2-dim numpy array.
    :param index: the index of the word that is to be similar.
    :param num: number of words to be returned
    """
    target_vector = feature_table[index]
    # calculate the similarity to each vector in the table
    similarities = np.apply_along_axis(cosine_similarity, 1, 
                                       feature_table, target_vector)
    
    # now get the last indices
    top_indices = similarities.argsort()[-num:]
    
    return top_indices

if __name__ == '__main__':
    
    parser = ArgumentParser(description="This script will search the word dictionary and \
    list the words most similar to the given one.")
    parser.add_argument('word', type=str, help='Word to compare with', default=None)
    parser.add_argument('-n', help='Number of similar words', type=int, default=5, dest='num')
    args = parser.parse_args()
    
    target_word = unicode(args.word, 'utf-8')
    
    features = utils.load_features_from_file(config.FILES['type_features'])
    word_dict = load_word_dict()
    
    index = word_dict[target_word]
    similar_words_indices = find_similar_words(features, index, args.num)
    words = word_dict.get_words(similar_words_indices)
    
    for word in words:
        print word
    
    
    