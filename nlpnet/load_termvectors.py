# -*- coding: utf-8 -*-

"""
Script for reading a text file created with semanticvectors package and saving
in the format used by nlpnet.

The termvectors file must be in text format. Lucene .bin files can be converted with:
java pitt.search.semanticvectors.VectorStoreTranslater -lucenetotext termvectors.bin termvectors.txt
"""

import re
import cPickle
import argparse
import numpy as np

import config
import utils
from word_dictionary import WordDictionary

def convert_termvectors(input_file=config.FILES['termvectors'],
                        output_file=config.FILES['type_features']):
    """
    Reads the termvectors file and converts it to the format used by nlpnet.
    """
    wd = WordDictionary.init_empty()
    # the next value to be inserted into the dictionary is one plus the current maximum
    next_val = max(wd.values()) + 1
    # list to store feature vectors efficiently without knowing how many there are
    feature_vectors = []
    
    with open(input_file) as f:
        
        for i, line in enumerate(f):
            uline = unicode(line, 'utf-8')
            
            if i == 0:
                num_features = int(re.search('-dimension ([0-9]+)', uline).group(1))
                
                # creates vectors for paddings and rare tokens
                feature_vectors.extend(utils.generate_feature_vectors(3, num_features, -0.1, 0.1))
                continue
            
            # termvectors fields are separated by the | character, but it is not escaped
            # so we must check if | was actually the token in the first field
            fields = uline.split('|')
            if len(fields) == num_features + 1:
                token = fields[0]
            else:
                num_token_fields = len(fields) - num_features
                token_fields = fields[:num_token_fields]
                token = '|'.join(token_fields)
            
            features = np.fromiter((float(x) for x in fields[-num_features:]),
                                    np.float,
                                    num_features)
            wd[token] = next_val
            feature_vectors.append(features)
            next_val += 1
            
    # now, convert the python list into a numpy array
    feature_table = np.array(feature_vectors)
    wd.check()
    
    with open(config.FILES['word_dict_dat'], 'wb') as f:
        cPickle.dump(wd, f, 2)
    
    utils.save_features_to_file(feature_table, output_file)
    
    return (wd, feature_table)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser('Script to load feature vectors produced by SemanticVectors package.')
    parser.add_argument('task', type=str, choices=['srl', 'srl_classify', 'srl_boundary', 
                                                   'srl_predicates', 'pos'])
    args = parser.parse_args()
    
    task = args.task.lower()
    
    if task == 'srl':
        output_file = config.FILES['type_features_1step']
    elif task == 'srl_classify':
        output_file = config.FILES['type_features_classify']
    elif task == 'srl_boundary':
        output_file = config.FILES['type_features_boundary']
    elif task == 'srl_predicates':
        output_file = config.FILES['type_features_srl_predicates']
    elif task == 'pos':
        output_file = config.FILES['type_features_pos']


    wd, features = convert_termvectors(output_file=output_file)
    
    
