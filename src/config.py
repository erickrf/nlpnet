# -*- coding: utf-8 -*- 

"""
Configuration data for the system.
"""

import os


DIRS = {'data': os.path.join('..', 'data'),
        'corpora': os.path.join('..', 'data', 'corpora'),
    }

    
FILES = {
     # cross-task data
     'word_dict_dat': os.path.join(DIRS['data'], 'word-dict.pickle'),
     'type_features': os.path.join(DIRS['data'], 'types-features.npy'),
     'termvectors': os.path.join(DIRS['data'], 'termvectors.txt'),
     
     # Language Model
     'network_lm': os.path.join(DIRS['data'], 'lm-network.npz'),
     'metadata_lm': os.path.join(DIRS['data'], 'metadata-lm.pickle'),
     'type_features_lm': os.path.join(DIRS['data'], 'types-features-lm.npy'),
     'caps_features_lm': os.path.join(DIRS['data'], 'caps-features-lm.npy'),
     'suffix_features_lm': os.path.join(DIRS['data'], 'suffix-features-lm.npy'),
     
     # Annotated corpora
     'conll': os.path.join(DIRS['corpora'], 'PBrConst.conll'),
     'conll_test': os.path.join(DIRS['corpora'], 'PBrConst_test.conll'),
     
     # POS
     'network_pos': os.path.join(DIRS['data'], 'pos-network.npz'),
     'pos_tag_dict': os.path.join(DIRS['data'], 'pos-tag-dict.pickle'),
     'suffixes': os.path.join(DIRS['data'], 'suffixes.txt'),
     'macmorpho_sentences': os.path.join(DIRS['data'], 'macmorpho-sents.pickle'),
     'macmorpho_train': os.path.join(DIRS['data'], 'macmorpho-train.pickle'),
     'macmorpho_test': os.path.join(DIRS['data'], 'macmorpho-test.pickle'),
     'metadata_pos': os.path.join(DIRS['data'], 'metadata-pos.pickle'),
     'type_features_pos': os.path.join(DIRS['data'], 'types-features-pos.npy'),
     'caps_features_pos': os.path.join(DIRS['data'], 'caps-features-pos.npy'),
     'suffix_features_pos': os.path.join(DIRS['data'], 'suffix-features-pos.npy'),
     
     # chunk
     'chunk_tag_dict': os.path.join(DIRS['data'], 'chunk-tag-dict.pickle'),

     # SRL
     'network_srl': os.path.join(DIRS['data'], 'srl-network.npz'),
     'network_srl_boundary': os.path.join(DIRS['data'], 'srl-id-network.npz'),
     'network_srl_classify': os.path.join(DIRS['data'], 'srl-class-network.npz'),
     'network_srl_predicates': os.path.join(DIRS['data'], 'srl-class-predicates.npz'),
     'srl_sentences': os.path.join(DIRS['data'], 'srl-sentences.pickle'),
     'srl_tag_dict': os.path.join(DIRS['data'], 'srl-iob-tag-dict.pickle'),
     'srl_iob_tag_dict': os.path.join(DIRS['data'], 'srl-iob-tag-dict.pickle'),
     'srl_boundary_tag_dict': os.path.join(DIRS['data'], 'srl-boundary-tag-dict.pickle'),
     'srl_classify_tag_dict': os.path.join(DIRS['data'], 'srl-classify-tag-dict.pickle'),
     'srl_predicates_tag_dict': os.path.join(DIRS['data'], 'srl-predicates-tag-dict.pickle'),
     'type_features_boundary': os.path.join(DIRS['data'], 'types-features-id.npy'),
     'caps_features_boundary': os.path.join(DIRS['data'], 'caps-features-id.npy'),
     'pos_features_boundary': os.path.join(DIRS['data'], 'pos-features-id.npy'),
     'chunk_features_boundary': os.path.join(DIRS['data'], 'chunk-features-id.npy'),
     'type_features_classify': os.path.join(DIRS['data'], 'types-features-class.npy'),
     'caps_features_classify': os.path.join(DIRS['data'], 'caps-features-class.npy'),
     'pos_features_classify': os.path.join(DIRS['data'], 'pos-features-class.npy'),
     'chunk_features_classify': os.path.join(DIRS['data'], 'chunk-features-class.npy'),
     'type_features_1step': os.path.join(DIRS['data'], 'types-features-1step.npy'),
     'caps_features_1step': os.path.join(DIRS['data'], 'caps-features-1step.npy'),
     'pos_features_1step': os.path.join(DIRS['data'], 'pos-features-1step.npy'),
     'chunk_features_1step': os.path.join(DIRS['data'], 'chunk-features-1step.npy'),
     'type_features_srl_predicates': os.path.join(DIRS['data'], 'types-features-preds.npy'),
     'caps_features_srl_predicates': os.path.join(DIRS['data'], 'caps-features-preds.npy'),
     'pos_features_srl_predicates': os.path.join(DIRS['data'], 'pos-features-preds.npy'),
     'metadata_srl': os.path.join(DIRS['data'], 'metadata-srl.pickle'),
     'metadata_srl_boundary': os.path.join(DIRS['data'], 'metadata-srl-boundary.pickle'),
     'metadata_srl_classify': os.path.join(DIRS['data'], 'metadata-srl-classify.pickle'),
     'metadata_srl_predicates': os.path.join(DIRS['data'], 'metadata-srl-predicates.pickle'),
     'srl_gold': os.path.join(DIRS['data'], 'srl-gold.txt')
     }



