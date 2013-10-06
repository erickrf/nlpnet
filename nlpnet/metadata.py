# -*- coding: utf-8 -*-

"""
This script contains the definition of the Metadata class.
It can also be invoked in order to create a Metada object
and save it to a file in the data directory.
"""


import cPickle
import os

import config

class Metadata(object):
    """
    Class for storing metadata about a neural network and its 
    parameter files.
    """
    
    def __init__(self, task, use_caps=True, use_suffix=False, use_pos=False, use_chunk=False, use_lemma=False):
        self.task = task
        self.use_caps = use_caps
        self.use_suffix = use_suffix
        self.use_pos = use_pos
        self.use_chunk = use_chunk
        self.use_lemma = use_lemma
        self.metadata = 'metadata_%s' % task
        self.network = 'network_%s' % task
        
        if task != 'lm':
            self.tag_dict = '%s_tag_dict' % task
        else:
            self.tag_dict = None
        
        if task == 'srl_boundary':
            self.pred_dist_table = 'pred_dist_table_boundary'
            self.target_dist_table = 'target_dist_table_boundary'
            self.transitions = 'srl_transitions_boundary'
            self.type_features = 'type_features_boundary'
            self.caps_features = 'caps_features_boundary'
            self.pos_features = 'pos_features_boundary'
            self.chunk_features = 'chunk_features_boundary'
            self.suffix_features = None
            
        elif task == 'srl_classify':
            self.pred_dist_table = 'pred_dist_table_classify'
            self.target_dist_table = 'target_dist_table_classify'
            self.transitions = None
            self.type_features = 'type_features_classify'
            self.caps_features = 'caps_features_classify'
            self.pos_features = 'pos_features_classify'
            self.chunk_features = 'chunk_features_classify'
            self.suffix_features = None
        
        elif task == 'srl':
            # one step srl
            self.pred_dist_table = 'pred_dist_table_1step'
            self.target_dist_table = 'target_dist_table_1step'
            self.transitions = 'srl_transitions_1step'
            self.type_features = 'type_features_1step'
            self.caps_features = 'caps_features_1step'
            self.pos_features = 'pos_features_1step'
            self.chunk_features = 'chunk_features_1step'
            self.suffix_features = None
        
        else:
            self.type_features = 'type_features_%s' % task
            self.caps_features = 'caps_features_%s' % task
            self.pos_features = 'pos_features_%s' % task
            self.chunk_features = 'chunk_features_%s' % task
            self.suffix_features = 'suffix_features_%s' % task
    
    def __str__(self):
        """
        Shows the task at hand and which attributes are used.
        """
        lines = []
        lines.append("Metadata for task %s" % self.task)
        for k in self.__dict__:
            if isinstance(k, str) and k.startswith('use_'):
                lines.append('%s: %s' % (k, self.__dict__[k]))
        
        return '\n'.join(lines)
    
    def save_to_file(self): 
        """
        Save the contents of the metadata to a file. The filename is determined according
        to the task.
        """
        filename = 'metadata-%s.pickle' % self.task.replace('_', '-')
        filename = os.path.join(config.data_dir, filename)
        with open(filename, 'wb') as f:
            cPickle.dump(self.__dict__, f, 2)
    
    @classmethod
    def load_from_file(cls, task):
        """
        Reads the file containing the metadata for the given task and returns a 
        Metadata object.
        """
        # the actual content of the file is the __dict__ member variable, which contain all
        # the instance's data
        filename = os.path.join(config.data_dir, 
                                'metadata-%s.pickle' % task.replace('_', '-'))
        md = Metadata(None)
        with open(filename, 'rb') as f:
            data = cPickle.load(f)
        
        md.__dict__.update(data)
        
        return md



if __name__ == '__main__':
    
    import argparse
    
    parser = argparse.ArgumentParser(description='This script will save a metadata file in the data directory.')
    parser.add_argument('--task', choices=['srl', 'pos'], 
                        help='Task for which the network should be used.', type=str)
    parser.add_argument('--pos', help='Use POS as a feature',
                        action='store_true')
    parser.add_argument('--chunk', help='Use chunks as a feature',
                        action='store_true')
    parser.add_argument('--lemma', help='Use lemmas instead of the actual words',
                        action='store_true')
    parser.add_argument('--caps', help='Use capitalization as a feature',
                        action='store_true')
    parser.add_argument('--suf', help='Use suffix features', action='store_true', dest='suffix')
    parser.add_argument('--id', help='Only argument identification (SRL only)',
                        action='store_true', dest='identify')
    parser.add_argument('--class', help='Only argument classification (SRL only)',
                        action='store_true', dest='classify')
    parser.add_argument('--pred', help='Only predicate identification (SRL only)',
                        action='store_true', dest='predicates')
    
    args = parser.parse_args()
    if args.identify:
        args.task = 'srl_boundary'
    elif args.classify:
        args.task = 'srl_classify'
    elif args.predicates:
        args.task = 'srl_predicates'
    
    m = Metadata(args.task, args.caps, args.suffix, args.pos, args.chunk, args.lemma)
    m.save_to_file()
    
    
    
    
    
