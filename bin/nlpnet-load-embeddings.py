#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to load word embeddings from different representations
and save them in the nlpnet format (numpy arrays and a text
vocabulary).
"""

import argparse
import os
import re
import logging
import numpy as np
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
            word = unicode(line, 'utf-8').strip()
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
    import cPickle
    
    with open(filename, 'rb') as f:
        model = cPickle.load(f)
    matrix = model.get_word_embeddings()

    # remove <s>, </s> and <padding>
    matrix = np.append([matrix[0]], matrix[4:], axis=0)
    new_vectors = nlpnet.utils.generate_feature_vectors(2,
                                                        matrix.shape[1])
    matrix = np.append(matrix, new_vectors, axis=0)
    return matrix


def read_skipdep_embeddings(filename):
    """
    Load the feature matrix and vocabulary used by Bansal et al.
    for dependency parsing.
    """
    model = {}
    with open(filename, 'rb') as f:
        for line in f:
            fields = line.split()
            word = unicode(fields[0], 'utf-8')
            vector = np.fromiter((float(x) for x in fields[1:]), 
                                 dtype=np.float)
            model[word] = vector
    
    # group vectors by their corresponding words' normalized form
    clusters = defaultdict(list)
    for word, vector in model.iteritems():
        if '_<CH>' in word or word == '</s>' or word == '<ROOT>':
            # dependency relations; we're not interested in them
            # or apparently sentence end 
            continue
        
        if word.strip() == '':
            # there was a unicode control char in the vocabulary
            continue
        
        normalized_word = re.sub(r'\d', '9', word.lower())
        clusters[normalized_word].append(vector)
    
    # now, average out each cluster
    for word, vectors in clusters.iteritems():
        clusters[word] = np.mean(vectors, 0)
    
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
    model = gensim.models.Word2Vec.load(filename)
    matrix = model.syn0
    vocab_size, num_features = matrix.shape

    # create 2 vectors to represent the padding and one for rare words, 
    # if there isn't one already
    num_extra_vectors = 2 if '*RARE*' in model else 3
    extra_vectors = nlpnet.utils.generate_feature_vectors(num_extra_vectors, num_features)
    matrix = np.concatenate((matrix, extra_vectors))
    
    vocab = model.vocab
    # gensim saves words in UTF-8. We convert to unicode here for consistency
    # with the rest of the script
    sorted_words = [unicode(word, 'utf-8') 
                    for word in sorted(vocab, key=lambda x: vocab[x].index)]
    
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
    
    single - a single plain text file containing one word per line, followed 
        by its vectors. Everything is separated by whitespaces.
        This format also handles some special tokens used in skipdep by 
        Mohit Bansal et al. (2014).
        '''
    
    parser = argparse.ArgumentParser(description=__doc__, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('type', help='Format of the embeddings. See the description below.', 
                        choices=['plain', 'senna', 'gensim', 'word2embeddings', 'single'])
    parser.add_argument('embeddings', help='File containing the actual embeddings')
    parser.add_argument('-v', help='Vocabulary file, if applicable. '\
                        'In SENNA, it is hash/words.lst', dest='vocabulary')
    parser.add_argument('-o', help='Directory to save the output', default='.',
                        dest='output_dir')
    parser.add_argument('--task', help='Task for which the embeddings will be used. '\
                        'It determines the name of the embeddings file. If not given, '\
                        'it will be nlpnet-embeddings.npy.', dest='task', default=None, 
                        choices=['pos', 'srl', 'srl_boundary', 'unlabeled_dependency',
                                 'labeled_dependency', 'srl_classify', 'srl_predicates'])
    args = parser.parse_args()
    
    nlpnet.set_data_dir(args.output_dir)
    output_vocabulary = nlpnet.config.FILES['vocabulary']
    if args.task is None:
        output_embeddings = os.path.join(args.output_dir, 'nlpnet-embeddings.npy')
    else:
        key = 'type_features_%s' % args.task
        output_embeddings = nlpnet.config.FILES[key]
    
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
        
    wd = nlpnet.word_dictionary.WordDictionary.init_from_wordlist(words)
    
    if args.type == 'senna':
        # add a copy of the left hand padding to be the right hand padding
        # (padding right was the left item to be added)
        index_left = wd.index_padding_left
        index_right = wd.index_padding_right
        array = matrix[index_left]
        matrix = np.vstack((matrix, array))
    
    logger.info('Saving nlpnet data...')
    np.save(output_embeddings, matrix)
    wd.save(output_vocabulary)
