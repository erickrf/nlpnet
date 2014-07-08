#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to load word embeddings from different representations
and save them in the nlpnet format (numpy arrays and a text
vocabulary).
"""

import argparse
import os
import logging
import numpy as np

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
    index_padding = words.index('PADDING')
    index_rare = words.index('UNKNOWN')
    
    WD = nlpnet.word_dictionary.WordDictionary
    words[index_padding] = WD.padding_left
    words[index_rare] = WD.rare
    words.append(WD.padding_right)
    
    return words


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
        '''
    
    parser = argparse.ArgumentParser(description=__doc__, epilog=epilog,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('type', help='Format of the embeddings. See the description below.', 
                        choices=['senna', 'plain'])
    parser.add_argument('embeddings', help='File containing the actual embeddings')
    parser.add_argument('-v', help='Vocabulary file, if applicable. '\
                        'In SENNA, it is hash/words.lst', dest='vocabulary')
    parser.add_argument('-o', help='Directory to save the output', default='.',
                        dest='output_dir')
    parser.add_argument('--task', help='Task for which the embeddings will be used. '\
                        'It determines the name of the embeddings file. If not given, '\
                        'it will be nlpnet-embeddings.npy.', dest='task', default=None, 
                        choices=['pos', 'srl', 'srl_boundary',
                                 'srl_classify', 'srl_predicates'])
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
