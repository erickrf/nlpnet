#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to load word embeddings from different representations
and save them in the nlpnet format (numpy arrays and a text
vocabulary).
"""

import argparse
import re
import logging
import numpy as np
from six.moves import cPickle
from collections import defaultdict

import nlpnet


def read_plain_embeddings(filename):
    """
    Read an embedding from a plain text file with one vector per 
    line, values separated by whitespace.
    """
    with open(filename, 'rb') as f:
        text = f.read().strip()
    
    text = text.replace('\r\n', '\n')
    lines = text.split('\n')
    matrix = np.array([[float(value) for value in line.split()]
                       for line in lines])
    
    return matrix


def read_plain_vocabulary(filename):
    """
    Read a vocabulary file containing one word type per line.
    Return a list of word types.
    """
    words = []
    with open(filename, 'rb') as f:
        for line in f:
            word = line.decode('utf-8').strip()
            if not word:
                continue
            words.append(word)
    
    return words


def read_senna_vocabulary(filename):
    """
    Read the vocabulary file used by SENNA. It has one word type per line,
    all lower case except for the special words PADDING and UNKNOWN.
    
    This function replaces these special words by the ones used in nlpnet.
    """
    words = read_plain_vocabulary(filename)
    
    # senna replaces all digits for 0, but nlpnet uses 9
    for i, word in enumerate(words):
        words[i] = word.replace('0', '9')
    
    index_padding = words.index('PADDING')
    index_rare = words.index('UNKNOWN')
    
    WD = nlpnet.word_dictionary.WordDictionary
    words[index_padding] = WD.padding_left
    words[index_rare] = WD.rare
    words.append(WD.padding_right)
    
    return words


def read_w2e_vocabulary(filename):
    """
    Read the vocabulary used with word2embeddings. It is the same as a plain
    text vocabulary, except the embeddings for the rare/unknown word correspond
    to the first vocabulary item (before the word in the actual file).
    """
    words = read_plain_vocabulary(filename)
    words.insert(0, nlpnet.word_dictionary.WordDictionary.rare)
    return words


def read_w2e_embeddings(filename):
    """
    Load the feature matrix used by word2embeddings.
    """
    with open(filename, 'rb') as f:
        model = cPickle.load(f)
    matrix = model.get_word_embeddings()

    # remove <s>, </s> and <padding>
    matrix = np.append([matrix[0]], matrix[4:], axis=0)
    new_vectors = nlpnet.utils.generate_feature_vectors(2,
                                                        matrix.shape[1])
    matrix = np.append(matrix, new_vectors, axis=0)
    return matrix


def read_polyglot_embeddings(filename):
    """
    Read vocabulary and embeddings from a file from polyglot.
    """
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    
    # first four words are UNK, <s>, </s> and padding
    # we discard <s> and </s>
    words = data[0]
    matrix = data[1].astype(np.float)
    matrix = np.delete(matrix, [1, 2], 0)
    
    WD = nlpnet.word_dictionary.WordDictionary
    words = [WD.rare, WD.padding_left] + list(words[4:])
    
    model = dict(zip(words, matrix))
    clusters = clusterize_words(model)
    
    vocabulary = clusters.keys()
    vocabulary.append(WD.padding_right)
    matrix = np.array(clusters.values())
    
    return matrix, vocabulary

    
def clusterize_words(model, filter_=None):
    """
    Group words in equivalent forms (e.g., with lower and uppercase letters)
    in clusters, then average out the vectors for each cluster.
    
    :param model: dictionary mapping words to vectors
    :param filter_: function that filters out words to be ignored
    """
    if filter_ is None:
        filter_ = lambda x: False
    
    # group vectors by their corresponding words' normalized form
    clusters = defaultdict(list)
    for word, vector in model.items():
        if filter_(word) or word.strip() == '':
            continue
        
        normalized_word = re.sub(r'\d', '9', word.lower())
        clusters[normalized_word].append(vector)
    
    # now, average out each cluster
    for word, vectors in clusters.items():
        clusters[word] = np.mean(vectors, 0)
    
    return clusters


def read_skipdep_embeddings(filename):
    """
    Load the feature matrix and vocabulary used by Bansal et al.
    for dependency parsing.
    """
    model = {}
    with open(filename, 'rb') as f:
        for line in f:
            fields = line.split()
            if len(fields) == 2:
                # some files have [num_words, vector_size] in the first line
                continue

            word = fields[0].decode('utf-8')
            vector = np.fromiter((float(x) for x in fields[1:]), 
                                 dtype=np.float)
            model[word] = vector
    
    def dep_filter(word):
        return '_<CH>' in word or word == '</s>' or word == '<ROOT>'
    
    clusters = clusterize_words(model, dep_filter)
    
    # and separate vocabulary from vectors
    vocabulary = clusters.keys()
    matrix = np.array(clusters.values())

    if '*unknown*' in clusters:
        # symbol used in skipdep
        index_rare = vocabulary.index('*unknown*')
        vocabulary[index_rare] = nlpnet.word_dictionary.WordDictionary.rare
    
    return matrix, vocabulary


def read_gensim_embeddings(filename):
    """
    Load the feature matrix used by gensim.
    """
    import gensim
    model = gensim.models.KeyedVectors.load_word2vec_format(filename)
    matrix = model.syn0
    vocab_size, num_features = matrix.shape

    # create 2 vectors to represent the padding and one for rare words, 
    # if there isn't one already
    num_extra_vectors = 2 if '*RARE*' in model else 3
    extra_vectors = nlpnet.utils.generate_feature_vectors(num_extra_vectors, num_features)
    matrix = np.concatenate((matrix, extra_vectors))
    
    vocab = model.vocab
    sorted_words = sorted(vocab, key=lambda x: vocab[x].index)
    
    return matrix, sorted_words


if __name__ == '__main__':
    
    epilog = '''
This script can deal with the following formats:
    
    plain - a plain text file containing one vector per line
        and a vocabulary file containing one word per line.
        It should NOT include special tokens for padding or 
        unknown words.
        
    senna - the one used by the SENNA system by Ronan Collobert.
        Same as plain, except it includes special words PADDING
        and UNKNOWN.
    
    gensim - a file format saved by the gensim library. It doesn't
        have a separate vocabulary file.
    
    word2embeddings - format used by the neural language model from
        word2embeddings (used in polyglot).
    
    polyglot - format of the files available from polyglot. It doesn't
        have a separate vocabulary file.
    
    single - a single plain text file containing one word per line, followed 
        by its vectors. Everything is separated by whitespaces.
        This format also handles some special tokens used in skipdep by 
        Mohit Bansal et al. (2014).
        '''

    class_ = argparse.RawDescriptionHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, epilog=epilog,
                                     formatter_class=class_)
    parser.add_argument('type', help='Format of the embeddings. '
                                     'See the description below.',
                        choices=['plain', 'senna', 'gensim', 'word2embeddings',
                                 'single', 'polyglot'])
    parser.add_argument('embeddings',
                        help='File containing the actual embeddings')
    parser.add_argument('-v', help='Vocabulary file, if applicable. '\
                        'In SENNA, it is hash/words.lst', dest='vocabulary')
    parser.add_argument('-o', help='Directory to save the output', default='.',
                        dest='output_dir')
    args = parser.parse_args()
    
    nlpnet.set_data_dir(args.output_dir)
    output_vocabulary = nlpnet.config.FILES['vocabulary']
    output_embeddings = nlpnet.config.FILES['type_features']
    
    nlpnet.utils.set_logger(logging.INFO)
    logger = logging.getLogger('Logger')
    logger.info('Loading data...')
    if args.type == 'senna':
        words = read_senna_vocabulary(args.vocabulary)
        matrix = read_plain_embeddings(args.embeddings)
    elif args.type == 'plain':
        words = read_plain_vocabulary(args.vocabulary)
        matrix = read_plain_embeddings(args.embeddings)
    elif args.type == 'gensim':
        matrix, words = read_gensim_embeddings(args.embeddings)
    elif args.type == 'word2embeddings':
        words = read_w2e_vocabulary(args.vocabulary)
        matrix = read_w2e_embeddings(args.embeddings)
    elif args.type == 'single':
        matrix, words = read_skipdep_embeddings(args.embeddings)
    elif args.type == 'polyglot':
        matrix, words = read_polyglot_embeddings(args.embeddings)
        
    wd = nlpnet.word_dictionary.WordDictionary.init_from_wordlist(words)
    
    if args.type in ('senna', 'polyglot'):
        # add a copy of the left hand padding to be the right hand padding
        # (padding right was the last item to be added)
        index_left = wd.index_padding_left
        array = matrix[index_left]
        matrix = np.vstack((matrix, array))
    
    logger.info('Saving nlpnet data...')
    np.save(output_embeddings, matrix)
    wd.save(output_vocabulary)
