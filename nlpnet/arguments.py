# -*- coding: utf-8 -*-

"""
Code for argument parsing and a few verifications. 
These arguments are used by the training script.
"""

import argparse


def fill_defaults(args, defaults_per_task):
    """
    This function fills arguments not explicitly set (left as None)
    with default values according to the chosen task.
    
    We can't rely on argparse to it because using subparsers with
    set_defaults and a parent parser overwrites the defaults. 
    """
    task = args.task
    defaults = defaults_per_task[task]
    for arg in args.__dict__:
        if getattr(args, arg) is None and arg in defaults:
            setattr(args, arg, defaults[arg])
    

def get_args():
    parser = argparse.ArgumentParser(description="Train nlpnet "\
                                     "for a given NLP task.")
    subparsers = parser.add_subparsers(title='Tasks',
                                       dest='task',
                                       description='Task to train nlpnet for. '\
                                       'Type %(prog)s [TASK] -h to get task-specific help.')
    
    defaults = {}
    
    # base parser with arguments not related to any model
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('-f', '--num_features', type=int,
                             help='Number of features per word '\
                             '(used to generate random vectors)',
                             default=50, dest='num_features')
    base_parser.add_argument('--load-network', action='store_true',
                             help='Load previously saved network')
    base_parser.add_argument('--load-features', action='store_true',
                             help="Load previously saved word type features "\
                             "(overrides -f and must also load a vocabulary file)", 
                             dest='load_types')
    base_parser.add_argument('-v', '--verbose', help='Verbose mode',
                             action="store_true")
    base_parser.add_argument('--gold', help='File with annotated data for training.', 
                             type=str, required=True)
    base_parser.add_argument('--data', help='Directory to save new models and load '\
                             'partially trained ones (default: current directory)', type=str, 
                             default='.')
    base_parser.add_argument('--dev', help='Development (validation) data. If not given, '\
                             'training data will be used to evaluate performance.',
                             default=None)
    
    # parser with network arguments shared among most tasks
    # each task-specific parser may define defaults
    network_parser = argparse.ArgumentParser(add_help=False, parents=[base_parser])
    
    network_parser.add_argument('-w', '--window', type=int,
                                help='Size of the word window',
                                dest='window')    
    network_parser.add_argument('-e', '--epochs', type=int, dest='iterations',
                                help='Number of training epochs')
    network_parser.add_argument('-l', '--learning_rate', type=float,
                                help='Learning rate for network connections',
                                dest='learning_rate')
    network_parser.add_argument('--l2', type=float, default=0,
                                help='L^2 regularization factor (aka lambda, default 0)')
    network_parser.add_argument('--dropout', type=float, default=0,
                                help='Dropout; probability to change a hidden value to 0 (default 0)')
    network_parser.add_argument('--max-norm', type=float, default=0, dest='max_norm',
                                help='Maximum weight norm; limits the norm of weight vectors (default 0 ie disabled)')
    network_parser.add_argument('-a', '--accuracy', type=float,
                                help='Maximum desired accuracy per token.',
                                default=0, dest='accuracy')
    network_parser.add_argument('-n', '--hidden', type=int,
                                help='Number of hidden neurons',
                                dest='hidden')
    network_parser.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                                help='Include capitalization features. '\
                                'Optionally, supply the number of features (default 5)')
    
        
    # parser with arguments shared among convolutional-based tasks
    conv_parser = argparse.ArgumentParser(add_help=False)
    conv_parser.add_argument('-c', '--convolution', type=int, 
                             help='Number of convolution neurons',
                             dest='convolution')
    conv_parser.add_argument('--pos', const=5, nargs='?', type=int, default=None,
                             help='Include part-of-speech features. '\
                             'Optionally, supply the number of features (default 5)')
    conv_parser.add_argument('--max-dist', type=int, default=10, dest='max_dist',
                             help='Maximum distance to have its own feature vector (default 10)')
    conv_parser.add_argument('--target-features', type=int, default=5, dest='target_features',
                             help='Number of features for distance to target word (default 5)')
    conv_parser.add_argument('--pred-features', type=int, default=5, dest='pred_features',
                             help='Number of features for distance to predicate (default 5)')
        
    
    # POS argument parser
    parser_pos = subparsers.add_parser('pos', help='POS tagging', 
                                       parents=[network_parser])
    parser_pos.add_argument('--suffix', const=2, nargs='?', type=int, default=None,
                            help='Include suffix features. Optionally, '\
                            'supply the number of features (default 2)')
    parser_pos.add_argument('--suffix_size', type=int, default=5,
                            help='Use suffixes up to this size (in characters, default 5). '\
                            'Only used if --suffix is supplied')
    parser_pos.add_argument('--prefix', const=2, nargs='?', type=int, default=None,
                            help='Include prefix features. Optionally, '\
                            'supply the number of features (default 2)')
    parser_pos.add_argument('--prefix_size', type=int, default=5,
                            help='Use prefixes up to this size (in characters, default 5). '\
                            'Only used if --suffix is supplied')
    defaults['pos'] = dict(window=5, hidden=100, iterations=15, 
                           learning_rate=0.001, learning_rate_features=0.001,
                           learning_rate_transitions=0.001)
    
    # dependency
    parser_dep = subparsers.add_parser('dependency', help='Dependency parsing')
    dep_subparsers = parser_dep.add_subparsers(title='Dependency parsing training steps',
                                               dest='subtask',
                                               description='Which step of the dependency training '\
                                               '(detecting edges or labeling them)')
    
    dep_subparsers.add_parser('labeled', help='Labeling dependency edges',
                              parents=[network_parser, conv_parser])
    dep_subparsers.add_parser('unlabeled', help='Dependency edge detection',
                              parents=[network_parser, conv_parser])
    
    defaults['dependency_filter'] = dict()
    defaults['labeled_dependency'] = dict(window=3)
    defaults['unlabeled_dependency'] = dict(window=3)
    
    # SRL argument parser
    # There is another level of subparsers for predicate detection / 
    # argument boundary identification / argument classification / 
    # (id + class) in one step
    parser_srl = subparsers.add_parser('srl', help='Semantic Role Labeling',
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser_srl.set_defaults(identify=False, predicates=False, classify=False)

    desc = '''SRL has 3 steps: predicate  detection, argument identification and 
argument classification. Each one depends on the one before.

You need one model trained for each subtask (or one for predicate
detection and another with the other 2 steps) in order to perform
full SRL.

Type %(prog)s [SUBTASK] -h to get subtask-specific help.'''
    
    srl_subparsers = parser_srl.add_subparsers(title='SRL subtasks',
                                               dest='subtask',
                                               description=desc)
    srl_subparsers.add_parser('pred', help='Predicate identification',
                              parents=[network_parser])
    defaults['srl_predicates'] = dict(window=5, hidden=50, iterations=1, 
                                      learning_rate=0.01, learning_rate_features=0.01,
                                      learning_rate_transitions=0.01,
                                      predicates=True)
    
    srl_subparsers.add_parser('id', help='Argument identification',
                              parents=[network_parser, conv_parser])
    defaults['srl_boundary'] = dict(window=3, hidden=150, convolution=150, 
                                    identify=True, iterations=15,
                                    learning_rate=0.001, learning_rate_features=0.001,
                                    learning_rate_transitions=0.001)
    
    srl_subparsers.add_parser('class', help='Argument classification',
                              parents=[network_parser, conv_parser])
    defaults['srl_classify'] = dict(window=3, hidden=0, convolution=100, 
                                    classify=True, iterations=3,
                                    learning_rate=0.01, learning_rate_features=0.01,
                                    learning_rate_transitions=0.01)
    srl_subparsers.add_parser('1step', parents=[network_parser, conv_parser],
                              help='Argument identification and '\
                              'classification together')
    defaults['srl'] = dict(window=3, hidden=150, convolution=200, iterations=15,
                           learning_rate=0.001, learning_rate_features=0.001,
                           learning_rate_transitions=0.001)
    
    
    args = parser.parse_args()
    
    if args.task == 'srl':
        if args.subtask == 'class':
            args.task = 'srl_classify'
            args.classify = True
        elif args.subtask == 'id':
            args.task = 'srl_boundary'
            args.identify = True
        elif args.subtask == 'pred':
            args.task = 'srl_predicates'
            args.predicates = True
    elif args.task == 'dependency':
        if args.subtask == 'labeled':
            args.task = 'labeled_dependency'
            args.labeled = True
        elif args.subtask == 'unlabeled':
            args.task = 'unlabeled_dependency'
            args.labeled = False
        
    fill_defaults(args, defaults)
    return args
