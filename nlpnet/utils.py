# -*- coding: utf-8 -*-

"""
Utility functions
"""

import re
import logging
import cPickle
import nltk
import numpy as np

from nltk.tokenize.regexp import RegexpTokenizer
import config
import attributes


def tokenize(text, clean=True):
    """
    Returns a list of lists of the tokens in text.
    Each line break in the text starts a new list.
    
    :param clean: If True, performs some cleaning action on the text, such as replacing
        numbers for the __NUMBER__ keyword.
    """
    ret = []
    
    if type(text) != unicode:
        text = unicode(text, 'utf-8')
    
    if clean:
        text = clean_text(text, correct=True)
    else:
        # replace numbers for __NUMBER__ and store them to replace them back
        numbers = re.findall(ur'\d+(?: \d+)*(?:[\.,]\d+)*[²³]*', text)
        numbers.extend(re.findall(ur'[²³]+', text))
        text = re.sub(ur'\d+( \d+)*([\.,]\d+)*[²³]*', '__NUMBER__', text)
        text = re.sub(ur'[²³]+', '__NUMBER__', text)
    
    # clitic pronouns
    regexp = r'''(?ux)
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
    text = re.sub(regexp, r'- \1', text)
    
    regexp = ur'''(?ux)
    # the order of the patterns is important!!
    ([^\W\d_]\.)+|                # one letter abbreviations, e.g. E.U.A.
    __NUMBER__:__NUMBER__|        # time and proportions
    [DSds][Rr][Aa]?\.|            # common abbreviations such as dr., sr., sra., dra.
    [^\W\d_]{1,2}\$|              # currency
    \w+([-']\w+)*-?|              # words with hyphens or apostrophes, e.g. não-verbal, McDonald's
                                  # or a verb with clitic pronoun removed (trailing hyphen is kept)
    -+|                           # any sequence of dashes
    \.{3,}|                       # ellipsis or sequences of dots
    __LINK__|                     # links found on wikipedia
    \S                            # any non-space character
    '''
    
    # loads trained model for tokenizing Portuguese sentences (provided by NLTK)
    sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    
    # the sentence tokenizer doesn't consider line breaks as sentence delimiters, so
    # we split them manually.
    sentences = []
    lines = text.split('\n')
    for line in lines:
        sentences.extend(sent_tokenizer.tokenize(line, realign_boundaries=True))
    
    t = RegexpTokenizer(regexp)
    
    for p in sentences:
        if p.strip() == '':
            continue
        
        # Wikipedia cleaning 
        if clean:
            # discard sentences with troublesome templates or links
            if any((x in p for x in ['__TEMPLATE__', '{{', '}}', '[[', ']]'])):
                continue
        
        new_sent = t.tokenize(p)
        
        if clean:
            # discard sentences that are a couple of words (it happens sometimes
            # when extracting data from lists).
            if len(new_sent) <= 2:
                continue
        elif len(numbers) > 0:
            # put back numbers that were previously replaced
            for i in xrange(len(new_sent)):
                token = new_sent[i]
                while '__NUMBER__' in token:
                    token = token.replace('__NUMBER__', numbers.pop(0), 1)
                new_sent[i] = token
        
        ret.append(new_sent)
        
    return ret

def clean_text(text, correct=True):
    """
    Apply some transformations to the text, such as mapping numbers to a __NUMBER__ keyword
    and simplifying quotation marks.
    
    :param correct: If True, tries to correct punctuation misspellings. 
    """
    
    # replaces different kinds of quotation marks with "
    # take care not to remove apostrophes
    text = re.sub(ur"(?u)(\W)[‘’′`']", r'\1"', text)
    text = re.sub(ur"(?u)[‘’`′'](\W)", r'"\1', text)
    text = re.sub(ur'(?u)[«»“”]', '"', text)
    
    if correct:
        # tries to fix mistyped tokens (common in Wikipedia-pt) as ,, '' ..
        text = re.sub(r'(?<!\.)\.\.(?!\.)', '.', text) # take care with ellipses 
        text = re.sub(r'([,";:])\1,', r'\1', text)
        
        # inserts space after leading hyphen. It happens sometimes in cases like:
        # blablabla -that is, bloblobloblo
        text = re.sub(' -(?=[^\W\d_])', ' - ', text)
    
    # replaces numbers with the __NUMBER__ token
    text = re.sub(ur'\d+( \d+)*([\.,]\d+)*[²³]*', '__NUMBER__', text)
    text = re.sub(ur'[²³]+', '__NUMBER__', text)
    
    # replaces special ellipsis character 
    text = re.sub(u'…', '...', text)
    
    return text



_contractible_base = ur'''(?iux)
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
_contractible_de = re.compile(_contractible_base % u'|aqui|aí|ali|entre')
_contractible_em = re.compile(_contractible_base % '')
_contractible_art = re.compile('[oa]s?')

def contract(w1, w2):
    """Makes a contraction of two words."""
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
            contraction = u'à' + w2[1:]
    elif w1 == 'para' and _contractible_art.match(w2):
        contraction = 'pr' + w2
    elif w1 == 'com':
        if w2 == 'mim':
            contraction = 'comigo'
        elif w2 == 'ti':
            contraction = 'contigo'
        elif w2 == 'si':
            contraction = 'consigo'
        elif w2 == u'nós':
            contraction = 'conosco'
        elif w2 == u'vós':
            contraction = 'convosco'
    elif w1 == 'lhe' and _contractible_art.match(w2):
        contraction = 'lh' + w2
    elif w1 == "d'":
        contraction = w1 + w2
    
    if contraction is None:
        raise ValueError('Unexpected contraction: "%s" and "%s"' % (w1, w2))
    
    return attributes.capitalize(contraction, cap) 

def generate_feature_vectors(num_vectors, num_features, min_value=-0.1, max_value=0.1):
    """
    Generates vectors of real numbers, to be used as word features.
    Vectors are initialized randomly. Returns a 2-dim numpy array.
    """
    logger = logging.getLogger("Logger")
    table = (max_value * 2) * np.random.random((num_vectors, num_features)) + min_value
    logger.debug("Generated %d feature vectors with %d features each." % (num_vectors,
                                                                          num_features))
    
    return table

def count_pos_tags():
    """Counts and returns how many POS tags there are."""
    with open(config.FILES['pos_tag_dict']) as f:
        td = cPickle.load(f)
    return len(td)

def count_chunk_tags():
    """Counts and returns how many chunk tags there are."""
    with open(config.FILES['chunk_tag_dict']) as f:
        td = cPickle.load(f)
    return len(td)

#TODO: this function could be more organized with less repeated code
def set_features(args, md, text_reader):
    """
    Loads the features to be used by the network. The actual number of 
    feature tables will depend on the argument options.
    
    :param arguments: Parameters supplied to the program
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
        types_table = load_features_from_file(config.FILES[md.type_features])
        
        if len(types_table) < len(text_reader.word_dict):
            # the type dictionary provided has more types than
            # the number of feature vectors. So, let's generate
            # feature vectors for the new types by replicating the vector
            # associated with the RARE word
            diff = len(text_reader.word_dict) - len(types_table)
            logger.warning("Number of types in feature table and dictionary differ.")
            logger.warning("Generating features for %d new types." % diff)
            num_features = len(types_table[0])
            new_vecs =  generate_feature_vectors(diff, num_features)
            types_table = np.append(types_table, new_vecs)
            
        elif len(types_table) < len(text_reader.word_dict):
            logger.warning("Number of features provided is greater than the number of tokens\
            in the dictionary. The extra features will be ignored.")
    
    feature_tables.append(types_table)
    
    # Capitalization
    if md.use_caps:
        # features for word capitalization
        # if the value is True, it means we should create new features. if the value is a 
        # string, then it is the name of the feature file
        if args.load_network:
            logger.info("Loading capitalization features...")
            caps_table = load_features_from_file(config.FILES[md.caps_features])
        else:
            logger.info("Generating capitalization features...")
            caps_table = generate_feature_vectors(attributes.Caps.num_values, args.caps)
        
        feature_tables.append(caps_table)
    
    # Suffixes
    if md.use_suffix:
        if args.load_network:
            logger.info("Loading suffix features...")
            suffix_table = load_features_from_file(config.FILES[md.suffix_features])
        else:
            logger.info("Generating suffix features...")
            suffix_table = generate_feature_vectors(attributes.Suffix.num_suffixes,
                                                    args.suffix)
        feature_tables.append(suffix_table)
    
    # POS tags
    if md.use_pos:
        if args.load_network:
            logger.info("Loading POS features...")
            pos_table = load_features_from_file(config.FILES[md.pos_features])
        else:
            logger.info("Generating POS features...")
            num_pos_tags = count_pos_tags()
            pos_table = generate_feature_vectors(num_pos_tags, args.pos)
    
        feature_tables.append(pos_table)
    
    # chunk tags
    if md.use_chunk:
        if args.load_network:
            logger.info("Loading chunk features...")
            chunk_table = load_features_from_file(config.FILES[md.chunk_features])
        else:
            logger.info("Generating chunk features...")
            num_chunk_tags = count_chunk_tags()
            chunk_table = generate_feature_vectors(num_chunk_tags, args.chunk)
        
        feature_tables.append(chunk_table)
    
    return feature_tables

def set_distance_features(md=None, max_dist=None, 
                          num_target_features=None, num_pred_features=None):
    """
    Returns the distance feature tables to be used by a convolutional network.
    One table is for relative distance to the target predicate, the other
    to the predicate.
    
    :param md: metadata containing the names of saved tables. Needed for 
        loading tables.
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

def set_logger(level):
    """Sets the logger to be used throughout the system."""
    log_format = '%(message)s'
    logging.basicConfig(format=log_format)
    logger = logging.getLogger("Logger")
    logger.setLevel(level)

def load_features_from_file(features_file):
    """Reads a file with features written as binary data."""
    return np.load(features_file)

def save_features_to_file(table, features_file):
    """Saves a feature table to a given file, writing binary data."""
    logger = logging.getLogger("Logger")
    np.save(features_file, table)
    logger.info('Saved %d vectors with %d features each to file %s' % 
                (table.shape[0], table.shape[1], features_file))
    
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


