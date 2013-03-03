# -*- coding: utf-8 -*-

"""
Auxiliary functions for POS tagging training and 
corpus reading.
"""

import re
import logging
from itertools import izip

import utils
from pos.macmorphoreader import MacMorphoReader

def create_reader_pos():
    """
    Creates and returns a TextReader object for the POS tagging task.
    """
    return MacMorphoReader()

def align_tags(sentences, tags):
    """
    Align a sequence of tags into a sequence of sequences, corresponding 
    to the sentences.
    """
    new_tags = []
    for sentence in sentences:
        sent_tags = []
        for _ in sentence:
            sent_tags.append(tags.pop(0))
        new_tags.append(sent_tags)
    
    return new_tags

def valid_sentence(sentence, tags):
    """
    Checks whether a sentence and its tags are valid or not.
    The check includes verifying if the sentence ends with an article
    or preposition, has a verb after an article, etc.
    """
    # first, check if the sentence ends with some punctuation sign
    # as some seem to be missing it
    last_tag = tags[-2] if tags[-1] == 'PU' else tags[-1]
    
    if last_tag in ('PREP', 'ART', 'PREP+ART'):
        # sentence ending with article or preposition
        return False
    
    # checking impossible sequences
    for token1, token2, tag1, tag2 in izip(sentence, sentence[1:], tags, tags[1:]):
        if tag1 in ('ART', 'PREP+ART') and tag2 in ('V', 'VAUX', 'PU'):
            return False
        
        if tag1 == tag2 == 'PU' and token1 == token2:
            # repeated punctuation
            return False
        
    return True

def read_macmorpho_file(filename):
    """
    Reads a file from the MacMorpho corpus and returns it as a list
    of sentences, where each sentence element is composed of a 
    tuple (token, tag).
    NOTE: Maybe it would be interesting to collapse the tags PRO-KS
    and PRO-KS-REL together, since they are very similar and distinguishing 
    them requires anaphora resolution 
    """
    import nltk.data
    st = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    
    tokens = []
    tags = []
    pre_token = ''
    pre_tag = ''
    waiting_contraction = False
    invalid = 0
    
    logger = logging.getLogger("Logger")
    
    with open(filename) as f:
        for line in f:
            uline = unicode(line, 'utf-8').strip()
            token, tag = uline.split('_')
            
            if waiting_contraction:
                if tag.endswith('|+'):
                    if tag[:-2] == pre_tag:
                        # the first part of the contraction continues. This happens
                        # in cases such as "Em frente de a", where Em, frente e de are 
                        # prepositions
                        tokens.append(pre_token)
                        tags.append(pre_tag)
                        pre_token = token
                        continue
                    else:
                        raise ValueError('Expected contraction between (%s, %s) and (%s, %s)'
                                         % (pre_token, pre_tag, token, tag))
                
                else:
                    # the contraction should be made now
                    
                    # I found there is one case of PREP+PREP which is dentre(de + entre)
                    # I think it's better to treat it as a simple prepostion
                    token = utils.contract(pre_token, token)
                    if not re.match('(?i)dentre', token):
                        tag = '%s+%s' % (pre_tag, tag)
                    waiting_contraction = False
                    tokens.append(token)
                    tags.append(tag)
                    continue
            
            if token == '$':
                # all cases of a single $ I found are mistakes
                continue
            elif re.match('\$\W', token):
                # there are some cases where a punctuation sign is preceded by $
                token = token[1:]
            
            if re.match('\W', tag):
                # normalize all punctuation tags
                tag = 'PU'
                
            elif re.match('V.*\|\+', tag):
                # append trailing hyphen to verbs with ênclise
                token += '-'
                tag = tag[:-2]
                
            elif tag == 'PREP|+':
                # wait for next token to contract
                waiting_contraction = True
                pre_tag = 'PREP'
                pre_token = token
                continue
            
            elif tag == 'NPROP|+':
                # cases too rare to care
                tag = 'NPROP'
            
            elif '|' in tag:
                # we will only use the complementar info after | for contractions
                # and ênclises, which are treated in other if blocks.
                # any other will be removed (including EST, DAT, HOR, etc.)
                tag = tag[:tag.find('|')]
            
            tokens.append(token)
            tags.append(tag)
    
    # This looks horrible but I haven't figured out a better way.
    # Join all the tokens as a string and have the sentence tokenizer 
    # split it into sentences. Then, align the sentences with the tags.
    # Then, verify weird sentences that should be put off along with
    # their tags.
    sents = st.tokenize(' '.join(tokens), realign_boundaries=False)
    sents = [sent.split() for sent in sents]
    tags = align_tags(sents, tags)
    
    tagged_sents = []
    valid_sents = []
    for sent, sent_tags in izip(sents, tags):
        # remove invalid and repeated sentences
        if valid_sentence(sent, sent_tags) and sent not in valid_sents:
            valid_sents.append(sent)
            tagged_sents.append(zip(sent, sent_tags))
             
        else:
            invalid += 1
    
    logger.debug('%d invalid sentence(s) discarded' % invalid)
    return tagged_sents

if __name__ == '__main__':
    pass



