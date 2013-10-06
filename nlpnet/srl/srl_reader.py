# -*- coding: utf-8 -*-

"""
Class for dealing with SRL data.
"""

from collections import defaultdict
import cPickle
import numpy as np
from itertools import izip

import config
import read_data
import attributes
from reader import TaggerReader
from utils import clean_text

class SRLReader(TaggerReader):
    
    def __init__(self, sentences=None, filename=None, only_boundaries=False, 
                 only_classify=False, only_predicates=False):
        """
        If no sentences argument is given, the reader will read the PropBank
        CoNLL file. If it is given, no reading is necessary (which saves a lot
        of time).
        :param sentence: a list of tuples in the format (tokens, list of tags, 
        predicate indices).
        """
        if only_boundaries:
            self.task = 'srl_boundary'
        elif only_classify:
            self.task = 'srl_classify'
        elif only_predicates:
            self.task = 'srl_predicates'
        else:
            self.task = 'srl'
        self.rare_tag = 'O'
        
        if sentences is None:
            
            if filename is not None:
            
                with open(filename) as f:
                    sent_data = read_data.read_conll(f)
                sents = [x[0] for x in sent_data]
                tags = [x[1] for x in sent_data]
                self.sentences = zip(sents, tags)
                
                preds = [x[2] for x in sent_data]
                self.predicates = [np.array(x) for x in preds]
                self._clean_text()
                self._make_contractions()
            
        else:
            self.sentences, self.predicates = sentences
            
        self.codified = False
    
    def extend(self, data):
        """
        Adds more data to the reader.
        :param data: a list of tuples in the format (tokens, tags, predicates), 
        one for each sentence.
        """
        self.sentences.extend([(sent, tags) for sent, tags, _ in data])
        self.predicates.extend([np.array(preds) for _, _, preds in data])
    
    def _clean_text(self):
        """
        Cleans the sentences text, replacing numbers for a keyword, different
        kinds of quotation marks for a single one, etc.
        """
        for sent, _ in self.sentences:
            for i, token in enumerate(sent):
                new_word = clean_text(token.word, correct=False)
                new_lemma = clean_text(token.lemma, correct=False) 
                token.word = new_word
                token.lemma = new_lemma
                sent[i] = token
        
    
    def _make_contractions(self):
        """
        Makes preposition contractions in the input data. It will contract words
        likely to be contracted, but there's no way to be sure the contraction 
        actually happened in the corpus. 
        """
        def_articles = ['a', 'as', 'o', 'os']
        adverbs = [u'aí', 'aqui', 'ali']
        pronouns = ['ele', 'eles', 'ela', 'elas', 'esse', 'esses', 
                    'essa', 'essas', 'isso', 'este', 'estes', 'esta',
                    'estas', 'isto', ]
        pronouns_a = ['aquele', 'aqueles', 'aquela', 'aquelas', 'aquilo',]
        
        for (sent, props), preds in zip(self.sentences, self.predicates):
            for i, token in enumerate(sent):
                try:
                    next_token = sent[i + 1]
                    next_word = next_token.word
                except IndexError:
                    # we are already at the last word.
                    break
                
                # look at the arg types for this and the next token in all propostions
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
                
                # check if the tags for this token and the next are the same in all propositions
                # if the first is O, however, we will merge them anyway.
                if all(a1 == a2 or a1 == 'O' for a1, a2 in zip(arg_types, next_arg_types)):
                    
                    if word == 'de' and next_word in (def_articles + pronouns + pronouns_a + adverbs):
                        contract('d' + next_word, 'd' + next_token.lemma)
                    
                    elif word == 'em' and next_word in (def_articles + pronouns + pronouns_a):
                        contract('n' + next_word, 'n' + next_token.lemma)
                    
                    elif word == 'por' and next_word in def_articles:
                        contract('pel' + next_word, 'pel' + next_token.lemma)
                    
                    elif word == 'a':
                        if next_word in pronouns_a:
                            contract(u'à' + next_word[1:], u'à' + next_token.lemma[1:])
                        
                        elif next_word in ['o', 'os']:
                            contract('a' + next_word, 'ao')
                        
                        elif next_word == 'a':
                            contract(u'à', 'ao')
                        
                        elif next_word == 'as':
                            contract(u'às', 'ao')
                    
    def _find_predicates(self):
        """
        Finds the index of the predicate of each sentence.
        """
        self.predicates = []
        for _, props in self.sentences:
            sentence_preds = []
            
            for prop in props:
                pred = [i for i, tag in enumerate(prop) if tag == 'V']
                assert len(pred) == 1, 'Proposition with more than one predicate'
                sentence_preds.append(pred[0])
            
            self.predicates.append(np.array(sentence_preds))
    
    def create_converter(self, metadata):
        """
        This function overrides the TextReader's one in order to deal with Token
        objects instead of raw strings.
        """
        self.converter = attributes.TokenConverter()
        
        if metadata.use_lemma:
            # look up word lemmas 
            word_lookup = lambda t: self.word_dict.get(t.lemma)
        else:
            # look up the word itself
            word_lookup = lambda t: self.word_dict.get(t.word)
             
        self.converter.add_extractor(word_lookup)
        
        if metadata.use_caps:
            caps_lookup = lambda t: attributes.get_capitalization(t.word)
            self.converter.add_extractor(caps_lookup)
        
        if metadata.use_pos:
            with open(config.FILES['pos_tag_dict']) as f:
                pos_dict = cPickle.load(f)
                
            pos_def_dict = defaultdict(lambda: pos_dict['other'])
            pos_def_dict.update(pos_dict)
            pos_lookup = lambda t: pos_def_dict[t.pos]
            self.converter.add_extractor(pos_lookup)
        
        if metadata.use_chunk:
            with open(config.FILES['chunk_tag_dict']) as f:
                chunk_dict = cPickle.load(f)
            
            chunk_def_dict = defaultdict(lambda: chunk_dict['O'])
            chunk_def_dict.update(chunk_dict)
            chunk_lookup = lambda t: chunk_def_dict[t.chunk]
            self.converter.add_extractor(chunk_lookup)
    
    def generate_tag_dict(self):
        """
        Generates a tag dictionary, to convert the tag itself
        to an index to be used in the neural network.
        """
        self.tagset = set(tag
                          for _, props in self.sentences
                          for prop in props
                          for tag in prop)
        
        self.tag_dict = dict( zip( self.tagset,
                                   xrange(len(self.tagset))
                                   )
                             )
    
    def _remove_tag_names(self):
        """
        Removes the actual tag names, leaving only IOB or IOBES block delimiters.
        """
        for _, propositions in self.sentences:
            for tags in propositions:
                for i, tag in enumerate(tags):
                    tags[i] = tag[0]
    
    def _codify_sentences(self):
        """
        Internal helper function.
        """
        new_sentences = []
        self.tags = []
        
        for (sent, props), preds in izip(self.sentences, self.predicates):
            new_sent = []
            sentence_tags = []
            
            for token in sent:
                new_token = self.converter.convert(token)
                new_sent.append(new_token)
            
            if self.task == 'srl_predicates':    
                sentence_tags = np.zeros(len(sent), np.int)
                sentence_tags[preds] = 1
            else:
                for prop in props:
                    # for classifying arguments, leave the names. they will be changed later
                    if self.task == 'srl_classify':
                        prop_tags = prop
                    else:
                        prop_tags = np.array([self.tag_dict[tag] for tag in prop])
                    sentence_tags.append(prop_tags)
            
            new_sentences.append(np.array(new_sent))
            self.tags.append(sentence_tags)
        
        self.sentences = new_sentences
        self.codified = True
    
    def codify_sentences(self):
        """
        Converts each token in each sequence into indices to their feature vectors
        in feature matrices. The previous sentences as text are not accessible anymore.
        Tags are also encoded. This function takes care of the case of classifying 
        pre-delimited arguments.
        """
        self._codify_sentences()
        self.arg_limits = []
        
        if self.task == 'srl_classify':
            # generate the tags for each argument
            start = 0
            end = 0
            
            for i, propositions in enumerate(self.tags):
                new_sent_tags = []
                sent_args = []
                
                for prop_tags in propositions:
                    
                    new_prop_tags = []
                    prop_args = []
                    last_tag = 'O'
                    
                    for j, tag in enumerate(prop_tags):
                        if tag != last_tag:
                            # if we were inside an argument, it ended
                            # we may have started a new
                            if last_tag != 'O':
                                end = j - 1
                                prop_args.append(np.array([start, end]))
                            
                            if tag != 'O':
                                start = j
                                new_prop_tags.append(self.tag_dict[tag])
                            
                        last_tag = tag
                    else:
                        # after last iteration, check the last tag
                        if last_tag != 'O':
                            end = j
                            prop_args.append(np.array([start, end]))
                    
                    sent_args.append(np.array(prop_args))
                    new_sent_tags.append(np.array(new_prop_tags))
                
                self.arg_limits.append(sent_args)
                self.tags[i] = new_sent_tags
                     
    
    def convert_tags(self, scheme, update_tag_dict=True, only_boundaries=False):
        """
        Replaces each word label with an IOB or IOBES version, appending a prefix
        to them. 
        :param scheme: IOB or IOBES (In, Other, Begin, End, Single).
        :param update_dict: whether to update or not the tag dictionary after
        converting the tags.
        :param only_boundaries: if True, only leaves the IOBES tags and remove
        the actual tags.
        """
        scheme = scheme.lower()
        if scheme not in ('iob', 'iobes'):
            raise ValueError("Unknown tagging scheme: %s" % scheme)
        
        for _, props in self.sentences:
            for prop in props:
                
                last_tag = None
                for i, tag in enumerate(prop):
                    
                    if tag == 'O':
                        # O tag is independent from IBES
                        last_tag = tag 
                        continue
                    
                    try:
                        next_tag = prop[i + 1]
                    except IndexError:
                        # last word already
                        next_tag = None
                     
                    if tag != last_tag:
                        # a new block starts here. 
                        last_tag = tag
                        if scheme == 'iob' or next_tag == tag:
                            prop[i] = 'B-%s' % tag
                        else:
                            prop[i] = 'S-%s' % tag
                    else:
                        # the block continues. 
                        if scheme == 'iob' or next_tag == tag:
                            prop[i] = 'I-%s' % tag
                        else:
                            prop[i] = 'E-%s' % tag
            
        if only_boundaries:
            self._remove_tag_names()
        
        if update_tag_dict:
            self.generate_tag_dict()
        else:
            # treat any tag not appearing in the tag dictionary as O
            tagset = set(tag for _, props in self.sentences for prop in props for tag in prop)
            for tag in tagset:
                if tag not in self.tag_dict:
                    self.tag_dict[tag] = self.tag_dict['O']
    
    
if __name__ == '__main__':
    pass
    
    
    