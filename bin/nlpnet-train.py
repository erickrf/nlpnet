#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train a neural networks for NLP tagging tasks.

Author: Erick Rocha Fonseca
"""

import logging
import numpy as np

import nlpnet.config as config
import nlpnet.read_data as read_data
import nlpnet.utils as utils
import nlpnet.taggers as taggers
import nlpnet.metadata as metadata
import nlpnet.srl as srl
import nlpnet.pos as pos
import nlpnet.parse as parse
import nlpnet.arguments as arguments
import nlpnet.reader as reader
import nlpnet.attributes as attributes
from nlpnet.network import Network, ConvolutionalNetwork, DependencyNetwork


############################
### FUNCTION DEFINITIONS ###
############################

def create_reader(args, md, validation=False):
    """
    Creates and returns a TextReader object according to the task at hand.
    
    :param args: the object containing the program arguments 
    :param md: metadata for the task
    :param validation: whether the reader should read the validation data
        from `args`
    """
    if validation:
        msg = "validation"
        filename = args.dev
    else:
        msg = "training"
        filename = args.gold
    
    logger.info("Reading %s data..." % msg)
    if args.task == 'pos':
        text_reader = pos.pos_reader.POSReader(md, filename=filename)
        if args.suffix:
            text_reader.create_affix_list('suffix', args.suffix_size, 5)
        if args.prefix:
            text_reader.create_affix_list('prefix', args.prefix_size, 5)
            
    elif args.task.startswith('srl'):
        text_reader = srl.srl_reader.SRLReader(md, filename=filename, only_boundaries=args.identify, 
                                               only_classify=args.classify,
                                               only_predicates=args.predicates)
        if args.identify:
            # only identify arguments
            text_reader.convert_tags('iobes', only_boundaries=True)
            
        elif not args.classify and not args.predicates:
            # this is SRL as one step, we use IOB
            text_reader.convert_tags('iob', update_tag_dict=False)
    
    elif args.task.endswith('dependency'):
        text_reader = parse.parse_reader.DependencyReader(md, filename)
    
    else:
        raise ValueError("Unknown task: %s" % args.task)
    
    text_reader.load_or_create_dictionary()
    text_reader.load_or_create_tag_dict()
    text_reader.create_converter()
    text_reader.codify_sentences()
    
    return text_reader
    

def create_network(args, text_reader, feature_tables, md=None):
    """Creates and returns the neural network according to the task at hand."""
    logger = logging.getLogger("Logger")
    
    convolution_srl =  args.task.startswith('srl') and args.task != 'srl_predicates'
    convolution = convolution_srl or args.task.endswith('dependency')
    
    if convolution:
        # get some data structures used both by dep parsing and SRL
        distance_tables = utils.set_distance_features(args.max_dist, args.target_features,
                                                      args.pred_features)
        padding_left = text_reader.converter.get_padding_left(False)
        padding_right = text_reader.converter.get_padding_right(False)
    
        if args.task.endswith('dependency'):
            output_size = 1 if not args.labeled else len(text_reader.tag_dict)
            nn = DependencyNetwork.create_new(feature_tables, distance_tables[0], 
                                              distance_tables[1], args.window, 
                                              args.convolution, args.hidden, output_size)
    
        else:
            num_tags = len(text_reader.tag_dict)
            nn = ConvolutionalNetwork.create_new(feature_tables, distance_tables[0], 
                                                 distance_tables[1], args.window, 
                                                 args.convolution, args.hidden, num_tags)
            
            if args.identify:
                logger.info("Loading initial transition scores table for argument identification")
                transitions = srl.train_srl.init_transitions_simplified(text_reader.tag_dict)
                nn.transitions = transitions
                nn.learning_rate_trans = args.learning_rate_transitions
                
            elif not args.classify:
                logger.info("Loading initial IOB transition scores table")
                transitions = srl.train_srl.init_transitions(text_reader.tag_dict, 'iob')
                nn.transitions = transitions
                nn.learning_rate_trans = args.learning_rate_transitions
                
    else:
        num_tags = len(text_reader.tag_dict)
        nn = Network.create_new(feature_tables, args.window, args.hidden, num_tags)
        if args.learning_rate_transitions > 0:
            transitions = np.zeros((num_tags + 1, num_tags), np.float)
            nn.transitions = transitions
            nn.learning_rate_trans = args.learning_rate_transitions

        padding_left = text_reader.converter.get_padding_left(args.task == 'pos')
        padding_right = text_reader.converter.get_padding_right(args.task == 'pos')
    
    nn.padding_left = np.array(padding_left)
    nn.padding_right = np.array(padding_right)
    nn.learning_rate = args.learning_rate
    nn.learning_rate_features = args.learning_rate_features
    
    if 'convolution' in args and args.convolution > 0 and args.hidden > 0:
        layer_sizes = (nn.input_size, nn.hidden_size, nn.hidden2_size, nn.output_size)
    else:
        layer_sizes = (nn.input_size, nn.hidden_size, nn.output_size)
    
    logger.info("Created new network with the following layer sizes: %s"
                % ', '.join(str(x) for x in layer_sizes))
    
    return nn

def load_network_train(args, md):
    """Loads and returns a neural network with all the necessary data."""
    nn = taggers.load_network(md)
    
    logger.info("Loaded network with following parameters:")
    logger.info(nn.description())
    
    nn.learning_rate = args.learning_rate
    nn.learning_rate_features = args.learning_rate_features
    if md.task.startswith('srl') or md.task == 'pos':
        nn.learning_rate_trans = args.learning_rate_transitions
    
    return nn

def create_metadata(args):
    """Creates a Metadata object from the given arguments."""
    # using getattr because the SRL args object doesn't have a "suffix" attribute
    use_caps = getattr(args, 'caps', False)
    use_suffix = getattr(args, 'suffix', False)
    use_prefix = getattr(args, 'prefix', False)
    use_pos = getattr(args, 'pos', False)
    use_chunk = getattr(args, 'chunk', False)
    
    return metadata.Metadata(args.task, None, use_caps, use_suffix, use_prefix, 
                               use_pos, use_chunk)

def set_validation_data(nn, task, reader):
    """Sets the neural network validation data."""
    if task == 'pos' or task == 'srl_predicates':
        nn.set_validation_data(reader.sentences, reader.tags)
    
    elif task.startswith('srl') and task != 'srl_predicates':
        arg_limits = None if task != 'srl_classify' else reader.arg_limits
        nn.set_validation_data(reader.sentences, reader.predicates,
                               reader.tags, arg_limits)
    
    elif task.endswith('dependency'):
        labels = None if task.startswith('unlabeled') else reader.labels
        nn.set_validation_data(reader.sentences, reader.heads, labels)
    
    else:
        raise ValueError('Unknown task: %s' % task)

def train(nn, reader, args):
    """Trains a neural network for the selected task."""
    num_sents = len(reader.sentences)
    logger.info("Starting training with %d sentences" % num_sents)
    
    avg_len = sum(len(x) for x in text_reader.sentences) / float(num_sents)
    logger.debug("Average sentence length is %f tokens" % avg_len)
    
    logger.debug("Network connection learning rate: %f" % nn.learning_rate)
    logger.debug("Feature vectors learning rate: %f" % nn.learning_rate_features)
    logger.debug("Tag transition matrix learning rate: %f\n" % nn.learning_rate_trans)
    
    intervals = max(args.iterations / 200, 1)
    np.seterr(over='raise')
    
    if args.task.startswith('srl') and args.task != 'srl_predicates':
        arg_limits = None if args.task != 'srl_classify' else text_reader.arg_limits
        
        nn.train(reader.sentences, reader.predicates, reader.tags, 
                 args.iterations, intervals, args.accuracy, arg_limits)
    
    elif args.task.endswith('dependency'):
        labels = None if not args.labeled else text_reader.labels
        nn.train(reader.sentences, reader.heads, args.iterations, 
                 intervals, args.accuracy, labels)
        
    else:
        nn.train(reader.sentences, reader.tags, 
                 args.iterations, intervals, args.accuracy)



if __name__ == '__main__':
    args = arguments.get_args()

    logging_level = logging.DEBUG if args.verbose else logging.INFO
    utils.set_logger(logging_level)
    logger = logging.getLogger("Logger")

    config.set_data_dir(args.data)
    
    if not args.load_network:
        # if we are about to create a new network, create the metadata too
        md = create_metadata(args)
        md.save_to_file()
    else:
        md = metadata.Metadata.load_from_file(args.task)
        
    text_reader = create_reader(args, md)
    
    if args.load_network:
        logger.info("Loading provided network...")
        nn = load_network_train(args, md)
    else:
        logger.info('Creating new network...')
        feature_tables = utils.create_feature_tables(args, md, text_reader)
        nn = create_network(args, text_reader, feature_tables, md)
    
    if args.dev is not None:
        validation_reader = create_reader(args, md, True)
        set_validation_data(nn, args.task, validation_reader)
    
    nn.network_filename = config.FILES[md.network]
    
    if args.decay:
        nn.set_learning_rate_decay(args.decay)
    
    train(nn, text_reader, args)
    logger.info("Finished training")
    
