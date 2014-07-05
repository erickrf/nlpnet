# -*- coding: utf-8 -*-

"""
Script for argument parsing and a few verifications. 
These arguments used by the training script.
"""

import argparse

# Tasks performed: part-of-speech tagging and semantic role labeling
TASKS = set(['pos', 'srl', 'dependency'])

def get_args():
    parser = argparse.ArgumentParser(description="NLPNet performs NLP taks like "\
                                     "POS tagging, semantic role labeling and dependency "\
                                     "parsing. Refer to the documentation for more information.")
    
    parser.add_argument('-w', '--window', type=int,
                        help='Size of the word window',
                        default=5, dest='window')
    parser.add_argument('-f', '--num_features', type=int,
                        help='Number of features per word',
                        default=50, dest='num_features')
    parser.add_argument('--load_features', action='store_true',
                        help="Load previously saved word type features (overrides -f and must also \
                        load a dictionary file)", dest='load_types')
    parser.add_argument('-e', '--epochs', type=int,
                        help='Number of training epochs',
                        default=10, dest='iterations')
    parser.add_argument('-l', '--learning_rate', type=float,
                        help='Learning rate for network connections',
                        dest='learning_rate')
    parser.add_argument('--lf', type=float, default=0,
                        help='Learning rate for features',
                        dest='learning_rate_features')
    parser.add_argument('--lt', type=float, default=0,
                        help='Learning rate for transitions (SRL/POS only)',
                        dest='learning_rate_transitions')
    parser.add_argument('--caps', const=5, nargs='?', type=int, default=None,
                        help='Include capitalization features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--suffix', const=5, nargs='?', type=int, default=None,
                        help='Include suffix features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--prefix', const=5, nargs='?', type=int, default=None,
                        help='Include prefix features. Optionally, supply the number of features (default 5)')
    parser.add_argument('--pos', const=5, nargs='?', type=int, default=None,
                        help='Include part-of-speech features (for SRL only). Optionally, supply the number of features (default 5)')
    parser.add_argument('--chunk', const=5, nargs='?', type=int, default=None,
                        help='Include chunk features (for SRL only). Optionally, supply the number of features (default 5)')
    parser.add_argument('--use_lemma', action='store_true',
                        help='Use word lemmas instead of surface forms.', dest='use_lemma')
    parser.add_argument('-a', '--accuracy', type=float,
                        help='Desired accuracy per tag (or per token in dependency parsing).',
                        default=0, dest='accuracy')
    parser.add_argument('-n', '--hidden', type=int, default=100,
                        help='Number of hidden neurons',
                        dest='hidden')
    parser.add_argument('-c', '--convolution', type=int, default=0,
                        help='Number of convolution neurons',
                        dest='convolution')
    parser.add_argument('-v', '--verbose', help='Verbose mode',
                        action="store_true")
    parser.add_argument('--id', help='Identify argument boundaries (do not classify)', action='store_true',
                        dest='identify')
    parser.add_argument('--class', help='Classify previously delimited SRL arguments', action='store_true',
                        dest='classify')
    parser.add_argument('--pred', help='Only predicate identification (SRL only)',
                        action='store_true', dest='predicates')
    parser.add_argument('--load_network', action='store_true',
                        help='Load previously saved network')
    parser.add_argument('--max_dist', type=int, default=10,
                        help='Maximum distance to have a separate feature (SRL/Dep parsing only)')
    parser.add_argument('--target_features', type=int, default=5,
                        help='Number of features for distance to target word (SRL/Dep parsing only)')
    parser.add_argument('--pred_features', type=int, default=5,
                        help='Number of features for distance to predicate (SRL/Dep parsing only)')
    parser.add_argument('--task', help='Task for which the network should be used.',
                        type=str, choices=TASKS, required=True)
    parser.add_argument('--semi', help='Perform semi-supervised training. Supply the name of the file with automatically tagged data.',
                        type=str, default='')
    parser.add_argument('--gold', help='File with annotated data for training.', type=str, default=None)
    parser.add_argument('--data', help='Directory to save new models and load partially trained ones', type=str, default=None, required=True)
    
    args = parser.parse_args()
    
    return args
    

def check_arguments(args):
    """
    Checks for possible inconsistencies in the arguments. Emits warnings for some 
    situations and halts execution if need be.
    """
    if args.task == 'srl':
        if args.classify:
            args.task = 'srl_classify'
        elif args.identify: 
            args.task = 'srl_boundary'
        elif args.predicates:
            args.task = 'srl_predicates'
    
    
    return args

if __name__ == '__main__':
    pass
