# -*- coding: utf-8 -*-

"""
This script contains the definition of the Metadata class.
It can also be invoked in order to create a Metada object
and save it to a file in the data directory.
"""

from six.moves import cPickle

from nlpnet import config


class Metadata(object):
    """
    Class for storing metadata about a neural network and its 
    parameter files.
    """
    
    def __init__(self, task, paths=None, use_caps=True, use_suffix=False,
                 use_prefix=False, use_pos=False, use_chunk=False,
                 use_lemma=False):
        self.task = task
        self.paths = paths if paths else config.FILES
        self.use_caps = use_caps
        self.use_suffix = use_suffix
        self.use_prefix = use_prefix
        self.use_pos = use_pos
        self.use_chunk = use_chunk
        self.use_lemma = use_lemma
        self.metadata = 'metadata_%s' % task
        self.network = 'network_%s' % task
        self.tag_dict = '%s_tag_dict' % task
        
        # dependency edge filter doesn't use an actual neural network, so 
        # we call it "model" to be more consistent
        self.model = self.network
        
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
            self.prefix_features = 'prefix_features_%s' % task
    
    def __str__(self):
        """Shows the task at hand and which attributes are used."""
        lines = []
        lines.append("Metadata for task %s" % self.task)
        for k in self.__dict__:
            if isinstance(k, str) and k.startswith('use_'):
                lines.append('%s: %s' % (k, self.__dict__[k]))
        
        return '\n'.join(lines)
    
    def save_to_file(self): 
        """
        Save the contents of the metadata to a file. The filename is determined
        according to the task.
        """
        save_data = self.__dict__.copy()
        filename = self.paths['metadata_%s' % self.task]
        del(save_data['paths'])
        
        with open(filename, 'wb') as f:
            cPickle.dump(save_data, f, 2)
    
    @classmethod
    def load_from_file(cls, task, paths=None):
        """
        Reads the file containing the metadata for the given task and returns a 
        Metadata object.
        """
        if paths is None:
            paths = config.FILES
        md = Metadata(None, paths)
        
        # the actual content of the file is the __dict__ member variable,
        # which contain all the instance's data
        with open(paths['metadata_%s' % task], 'rb') as f:
            data = cPickle.load(f)
        md.__dict__.update(data)
        
        return md

