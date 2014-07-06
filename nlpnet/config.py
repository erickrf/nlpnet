# -*- coding: utf-8 -*- 

"""
Configuration data for the system.
"""

import os

data_dir = None
FILES = {}

def set_data_dir(directory):
    """Sets the data directory containing the data for the models."""
    assert os.path.isdir(directory), 'Invalid data directory'
    
    global data_dir, FILES
    data_dir = directory

    FILES = {
         # cross-task data
         'vocabulary': os.path.join(data_dir, 'vocabulary.txt'),
         'word_dict_dat': os.path.join(data_dir, 'vocabulary.txt'), # deprecated
         'type_features': os.path.join(data_dir, 'types-features.npy'),
         'termvectors': os.path.join(data_dir, 'termvectors.txt'),
         
         # POS
         'network_pos': os.path.join(data_dir, 'pos-network.npz'),
         'pos_tags': os.path.join(data_dir, 'pos-tags.txt'),
         'pos_tag_dict': os.path.join(data_dir, 'pos-tags.txt'),
         'suffixes': os.path.join(data_dir, 'suffixes.txt'),
         'prefixes': os.path.join(data_dir, 'prefixes.txt'),
         'metadata_pos': os.path.join(data_dir, 'metadata-pos.pickle'),
         'type_features_pos': os.path.join(data_dir, 'types-features-pos.npy'),
         'caps_features_pos': os.path.join(data_dir, 'caps-features-pos.npy'),
         'suffix_features_pos': os.path.join(data_dir, 'suffix-features-pos.npy'),
         'prefix_features_pos': os.path.join(data_dir, 'prefix-features-pos.npy'),
         
         # chunk
         'chunk_tag_dict': os.path.join(data_dir, 'chunk-tag-dict.pickle'),
         'chunk_tags': os.path.join(data_dir, 'chunk-tags.txt'),
    
         # SRL
         'network_srl': os.path.join(data_dir, 'srl-network.npz'),
         'network_srl_boundary': os.path.join(data_dir, 'srl-id-network.npz'),
         'network_srl_classify': os.path.join(data_dir, 'srl-class-network.npz'),
         'network_srl_predicates': os.path.join(data_dir, 'srl-class-predicates.npz'),
         'srl_iob_tag_dict': os.path.join(data_dir, 'srl-tags.txt'),
         'srl_iob_tags': os.path.join(data_dir, 'srl-tags.txt'),
         'srl_tags': os.path.join(data_dir, 'srl-tags.txt'),
         'srl_classify_tag_dict': os.path.join(data_dir, 'srl-tags.txt'),
         'srl_classify_tags': os.path.join(data_dir, 'srl-tags.txt'),
         'srl_predicates_tag_dict': os.path.join(data_dir, 'srl-predicates-tags.txt'),
         'srl_predicates_tags': os.path.join(data_dir, 'srl-predicates-tags.txt'),
         'type_features_boundary': os.path.join(data_dir, 'types-features-id.npy'),
         'caps_features_boundary': os.path.join(data_dir, 'caps-features-id.npy'),
         'pos_features_boundary': os.path.join(data_dir, 'pos-features-id.npy'),
         'chunk_features_boundary': os.path.join(data_dir, 'chunk-features-id.npy'),
         'type_features_classify': os.path.join(data_dir, 'types-features-class.npy'),
         'caps_features_classify': os.path.join(data_dir, 'caps-features-class.npy'),
         'pos_features_classify': os.path.join(data_dir, 'pos-features-class.npy'),
         'chunk_features_classify': os.path.join(data_dir, 'chunk-features-class.npy'),
         'type_features_1step': os.path.join(data_dir, 'types-features-1step.npy'),
         'caps_features_1step': os.path.join(data_dir, 'caps-features-1step.npy'),
         'pos_features_1step': os.path.join(data_dir, 'pos-features-1step.npy'),
         'chunk_features_1step': os.path.join(data_dir, 'chunk-features-1step.npy'),
         'type_features_srl_predicates': os.path.join(data_dir, 'types-features-preds.npy'),
         'caps_features_srl_predicates': os.path.join(data_dir, 'caps-features-preds.npy'),
         'pos_features_srl_predicates': os.path.join(data_dir, 'pos-features-preds.npy'),
         'metadata_srl': os.path.join(data_dir, 'metadata-srl.pickle'),
         'metadata_srl_boundary': os.path.join(data_dir, 'metadata-srl-boundary.pickle'),
         'metadata_srl_classify': os.path.join(data_dir, 'metadata-srl-classify.pickle'),
         'metadata_srl_predicates': os.path.join(data_dir, 'metadata-srl-predicates.pickle')
         }



