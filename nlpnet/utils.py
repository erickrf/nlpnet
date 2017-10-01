# -*- coding: utf-8 -*-

from __future__ import unicode_literals

"""
Utility functions
"""

import re
import os
import logging
import nltk
import numpy as np

from nltk.tokenize.regexp import RegexpTokenizer
from nltk.tokenize import TreebankWordTokenizer
from nlpnet import attributes

_tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    (?:[Mm]\.?[Ss][Cc])\.?|           # M.Sc. with or without capitalization and dots
    (?:[Pp][Hh]\.?[Dd])\.?|           # Same for Ph.D.
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    \d{1,3}(?:\.\d{3})*(?:,\d+)|      # numbers in format 999.999.999,99999
    \d{1,3}(?:,\d{3})*(?:\.\d+)|      # numbers in format 999,999,999.99999
    \d+:\d+|                          # time and proportions
    \d+(?:[-\\/]\d+)*|                # dates. 12/03/2012 12-03-2012
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    (?:[^\W\d_]){1,2}\$|              # currency
    (?:[\#@]\w+])|                    # Hashtags and twitter user names
    -(?:[^\W\d_])+|                   # clitic pronouns with leading hyphen
    \w+(?:[-']\w+)*|                  # words with hyphens or apostrophes, e.g. não-verbal, McDonald's
    -+|                               # any sequence of dashes
    \.{3,}|                           # ellipsis or sequences of dots
    \S                                # any non-space character
    '''
_tokenizer = RegexpTokenizer(_tokenizer_regexp)

# clitic pronouns
_clitic_regexp_str = r'''(?ux)
    (?<=\w)                           # a letter before
    -(me|
    te|
    o|a|no|na|lo|la|se|
    lhe|lho|lha|lhos|lhas|
    nos|
    vos|
    os|as|nos|nas|los|las|            # unless if followed by more chars
    lhes)(?![-\w])                    # or digits or hyphens
'''
_clitic_regexp = re.compile(_clitic_regexp_str)


def tokenize(text, language):
    """
    Call the tokenizer function for the given language.
    The returned tokens are in a list of lists, one for each sentence.
    
    :param language: two letter code (en, pt)
    """
    if language == 'en':
        return tokenize_en(text)
    elif language == 'pt':
        return tokenize_pt(text, False)


def tokenize_en(text):
    """
    Return a list of lists of the tokens in text, separated by sentences.
    """
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokenizer = TreebankWordTokenizer()
    sentences = [tokenizer.tokenize(sentence) 
                 for sentence in sent_tokenizer.tokenize(text)]
    return sentences

    
def tokenize_pt(text, clean=True):
    """
    Returns a list of lists of the tokens in text, separated by sentences.
    Each line break in the text starts a new list.
    
    :param clean: If True, performs some cleaning action on the text, such as
        replacing all digits for 9 (by calling :func:`clean_text`)
    """    
    if clean:
        text = clean_text(text, correct=True)
    
    text = _clitic_regexp.sub(r' -\1', text)
    
    # loads trained model for tokenizing Portuguese sentences (provided by NLTK)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    sentences = [_tokenizer.tokenize(sent)
                 for sent in sent_tokenizer.tokenize(text,
                                                     realign_boundaries=True)]
    
    return sentences


def clean_text(text, correct=True):
    """
    Apply some transformations to the text, such as 
    replacing digits for 9 and simplifying quotation marks.
    
    :param correct: If True, tries to correct punctuation misspellings. 
    """    
    # replaces different kinds of quotation marks with "
    # take care not to remove apostrophes
    text = re.sub(r"(?u)(^|\W)[‘’′`']", r'\1"', text)
    text = re.sub(r"(?u)[‘’`′'](\W|$)", r'"\1', text)
    text = re.sub(r'(?u)[«»“”]', '"', text)
    
    if correct:
        # tries to fix mistyped tokens (common in Wikipedia-pt) as ,, '' ..
        text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text) # take care with ellipses 
        text = re.sub(r'([,";:])\1,', r'\1', text)
        
        # inserts space after leading hyphen. It happens sometimes in cases like
        # blablabla -that is, bloblobloblo
        text = re.sub(' -(?=[^\W\d_])', ' - ', text)
    
    # replaces numbers with the 9's
    text = re.sub(r'\d', '9', text)
    
    # replaces special ellipsis character 
    text = text.replace('…', '...')
    
    return text


_contractible_base = r'''(?iux)
    (
    [ao]s?|                # definite articles
    um(as?)?|uns|          # indefinite articles
    is[st]o|aquilo|        # demonstratives
    es[st][ea]s?|
    aquel[ea]s?|
    el[ea]s?|              # personal pronouns
    outr[oa]s?
    %s    
    )
    $
    '''
_contractible_de = re.compile(_contractible_base % '|aqui|aí|ali|entre')
_contractible_em = re.compile(_contractible_base % '')
_contractible_art = re.compile('[oa]s?')


def contract(w1, w2):
    """
    Makes a contraction of two words (in Portuguese).

    For example: contract('de', 'os') returns 'dos'
    If a contraction between the given words doesn't exist in Portuguese, a
        ValueError
    exception is thrown.
    """
    cap = attributes.get_capitalization(w1)
    w1 = w1.lower()
    w2 = w2.lower()
    contraction = None
    
    if w1 == 'de' and _contractible_de.match(w2):
        contraction = 'd' + w2
    elif w1 == 'em' and _contractible_em.match(w2):
        contraction = 'n' + w2
    elif w1 == 'por' and _contractible_art.match(w2):
        contraction = 'pel' + w2
    elif w1 == 'a':
        if w2 in ['o', 'os']:
            contraction = 'a' + w2
        elif w2.startswith('a'):
            contraction = 'à' + w2[1:]
    elif w1 == 'para' and _contractible_art.match(w2):
        contraction = 'pr' + w2
    elif w1 == 'com':
        if w2 == 'mim':
            contraction = 'comigo'
        elif w2 == 'ti':
            contraction = 'contigo'
        elif w2 == 'si':
            contraction = 'consigo'
        elif w2 == 'nós':
            contraction = 'conosco'
        elif w2 == 'vós':
            contraction = 'convosco'
    elif w1 == 'lhe' and _contractible_art.match(w2):
        contraction = 'lh' + w2
    elif w1 == "d'":
        contraction = w1 + w2
    
    if contraction is None:
        raise ValueError('Unexpected contraction: "%s" and "%s"' % (w1, w2))
    
    return attributes.capitalize(contraction, cap)


def generate_feature_vectors(num_vectors, num_features, min_value=-0.1,
                             max_value=0.1):
    """
    Generates vectors of real numbers, to be used as word features.
    Vectors are initialized randomly. Returns a 2-dim numpy array.
    """
    logger = logging.getLogger("Logger")
    table = (max_value * 2) * np.random.random((num_vectors, num_features)) \
            + min_value
    base_msg = "Generated %d feature vectors with %d features each."
    logger.debug(base_msg % (num_vectors, num_features))
    
    return table


def count_lines(filename):
    """Counts and returns how many non empty lines in a file there are."""
    with open(filename, 'r') as f:
        lines = [x for x in list(f) if x.strip()]
    return len(lines)


def _create_affix_tables(affix, table_list, num_features):
    """
    Internal helper function for loading suffix or prefix feature tables 
    into the given list.
    affix should be either 'suffix' or 'prefix'.
    """
    logger = logging.getLogger('Logger')
    logger.info('Generating %s features...' % affix)
    tensor = []
    codes = getattr(attributes.Affix, '%s_codes' % affix)
    num_affixes_per_size = getattr(attributes.Affix,
                                   'num_%ses_per_size' % affix)
    for size in codes:
        
        # use num_*_per_size because it accounts for special suffix codes
        num_affixes = num_affixes_per_size[size]
        table = generate_feature_vectors(num_affixes, num_features)
        tensor.append(table)
    
    # affix attribute actually has a 3-dim tensor
    # (concatenation of 2d tables, one for each suffix size)
    for table in tensor:
        table_list.append(table)


def create_feature_tables(args, md, text_reader):
    """
    Create the feature tables to be used by the network. If the args object
    contains the load_features option as true, the feature table for word types
    is loaded instead of being created. The actual number of 
    feature tables will depend on the argument options.
    
    :param args: Parameters supplied to the program
    :param md: metadata about the network
    :param text_reader: The TextReader being used.
    :returns: all the feature tables to be used
    """
    
    logger = logging.getLogger("Logger")
    feature_tables = []
    
    if not args.load_types:
        logger.info("Generating word type features...")
        table_size = len(text_reader.word_dict)
        types_table = generate_feature_vectors(table_size, args.num_features)
    else:
        logger.info("Loading word type features...")
        # check if there is a word feature file specific for the task
        # if not, load a generic one
        filename = md.paths[md.type_features]
        if os.path.exists(filename):
            types_table = load_features_from_file(filename)
        else:
            filename = md.paths['type_features']
            types_table = load_features_from_file(filename)
        
        if len(types_table) < len(text_reader.word_dict):
            # the type dictionary provided has more types than
            # the number of feature vectors. So, let's generate
            # feature vectors for the new types by replicating the vector
            # associated with the RARE word
            diff = len(text_reader.word_dict) - len(types_table)
            logger.warning("Number of types in feature table and "
                           "dictionary differ.")
            logger.warning("Generating features for %d new types." % diff)
            num_features = len(types_table[0])
            new_vecs =  generate_feature_vectors(diff, num_features)
            types_table = np.append(types_table, new_vecs, axis=0)
            
        elif len(types_table) < len(text_reader.word_dict):
            logger.warning("Number of features provided is greater than the "
                           "number of tokens\
            in the dictionary. The extra features will be ignored.")
    
    feature_tables.append(types_table)
    
    # Capitalization
    if md.use_caps:
        logger.info("Generating capitalization features...")
        caps_table = generate_feature_vectors(attributes.Caps.num_values, args.caps)
        feature_tables.append(caps_table)
    
    # Prefixes
    if md.use_prefix:
        _create_affix_tables('prefix', feature_tables, args.prefix)
    
    # Suffixes
    if md.use_suffix:
        _create_affix_tables('suffix', feature_tables, args.suffix)
    
    # POS tags
    if md.use_pos:
        logger.info("Generating POS features...")
        num_pos_tags = text_reader.get_num_pos_tags()
        pos_table = generate_feature_vectors(num_pos_tags, args.pos)
        feature_tables.append(pos_table)
    
    # chunk tags
    if md.use_chunk:
        logger.info("Generating chunk features...")
        num_chunk_tags = count_lines(md.paths['chunk_tags'])
        chunk_table = generate_feature_vectors(num_chunk_tags, args.chunk)
        feature_tables.append(chunk_table)
    
    return feature_tables


def set_distance_features(max_dist=None, 
                          num_target_features=None, num_pred_features=None):
    """
    Returns the distance feature tables to be used by a convolutional network.
    One table is for relative distance to the target predicate, the other
    to the predicate.
    
    :param max_dist: maximum distance to be used in new vectors.
    """
    logger = logging.getLogger("Logger")
    
    # max_dist before/after, 0 distance, and distances above the max
    max_dist = 2 * (max_dist + 1) + 1
    logger.info("Generating target word distance features...")
    target_dist = generate_feature_vectors(max_dist, num_target_features)
    logger.info("Generating predicate distance features...")
    pred_dist = generate_feature_vectors(max_dist, num_pred_features)
    
    return [target_dist, pred_dist]


def make_contractions_srl(sentences, predicates):
    """
    Makes preposition contractions in the input data for SRL with Portuguese
    text.

    It will contract words likely to be contracted, but there's no way to be 
    sure the contraction actually happened in the corpus. 
    
    :param sentences: the sentences list used by SRLReader objects.
    :param predicates: the predicates list used by SRLReader objects.
    :returns: a tuple (sentences, predicates) after contractions have been made.
    """
    def_articles = ['a', 'as', 'o', 'os']
    adverbs = ['aí', 'aqui', 'ali']
    pronouns = ['ele', 'eles', 'ela', 'elas', 'esse', 'esses', 
                'essa', 'essas', 'isso', 'este', 'estes', 'esta',
                'estas', 'isto', ]
    pronouns_a = ['aquele', 'aqueles', 'aquela', 'aquelas', 'aquilo']
    
    for (sent, props), preds in zip(sentences, predicates):
        for i, token in enumerate(sent):
            try:
                next_token = sent[i + 1]
                next_word = next_token.word
            except IndexError:
                # we are already at the last word.
                break
            
            # look at the arg types for this and the next token in all
            # propostions
            arg_types = [prop[i] for prop in props]
            next_arg_types = [prop[i + 1] for prop in props]
            
            # store the type of capitalization to convert it back
            word = token.word.lower()
            cap = attributes.get_capitalization(token.word)
            
            def contract(new_word, new_lemma):
                token.word = attributes.capitalize(new_word, cap)
                token.lemma = new_lemma
                token.pos = '%s+%s' % (token.pos, next_token.pos)
                sent[i] = token
                del sent[i + 1]
                # removing a token will change the position of predicates
                preds[preds > i] -= 1
                for prop in props: del prop[i]
            
            # check if the tags for this token and the next are the same in all
            # propositions
            # if the first is O, however, we will merge them anyway.
            if all(a1 == a2 or a1 == 'O' for a1, a2 in zip(arg_types,
                                                           next_arg_types)):

                if word == 'de' and next_word in (def_articles + pronouns +
                                                  pronouns_a + adverbs):
                    contract('d' + next_word, 'd' + next_token.lemma)
                
                elif word == 'em' and next_word in (def_articles + pronouns +
                                                    pronouns_a):
                    contract('n' + next_word, 'n' + next_token.lemma)
                
                elif word == 'por' and next_word in def_articles:
                    contract('pel' + next_word, 'pel' + next_token.lemma)
                
                elif word == 'a':
                    if next_word in pronouns_a:
                        contract('à' + next_word[1:],
                                 'à' + next_token.lemma[1:])
                    
                    elif next_word in ['o', 'os']:
                        contract('a' + next_word, 'ao')
                    
                    elif next_word == 'a':
                        contract('à', 'ao')
                    
                    elif next_word == 'as':
                        contract('às', 'ao')
    
    return sentences, predicates


def set_logger(level):
    """Sets the logger to be used throughout the system."""
    log_format = '%(message)s'
    logging.basicConfig(format=log_format)
    logger = logging.getLogger("Logger")
    logger.setLevel(level)


def load_features_from_file(features_file):
    """Reads a file with features written as binary data."""
    return np.load(features_file, encoding='bytes')


def save_features_to_file(table, features_file):
    """Saves a feature table to a given file, writing binary data."""
    np.save(features_file, table)


def convert_iobes_to_bracket(tag):
    """
    Convert tags from the IOBES scheme to the CoNLL bracketing.
    
    Example:
    B-A0 -> (A0*
    I-A0 -> *
    E-A0 -> *)
    S-A1 -> (A1*)
    O    -> *
    """
    if tag.startswith('I') or tag.startswith('O'):
        return '*'
    if tag.startswith('B'):
        return '(%s*' % tag[2:]
    if tag.startswith('E'):
        return '*)'
    if tag.startswith('S'):
        return '(%s*)' % tag[2:]
    else:
        raise ValueError("Unknown tag: %s" % tag)


def boundaries_to_arg_limits(boundaries):
    """
    Converts a sequence of IOBES tags delimiting arguments to an array
    of argument boundaries, used by the network.
    """
    limits = []
    start = None
    
    for i, tag in enumerate(boundaries):
        if tag == 'S': 
            limits.append([i, i])
        elif tag == 'B':
            start = i 
        elif tag == 'E':
            limits.append([start, i])
    
    return np.array(limits, np.int)


