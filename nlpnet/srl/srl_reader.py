# -*- coding: utf-8 -*-

"""
Class for dealing with SRL data.
"""

from collections import defaultdict
import cPickle
import numpy as np
from itertools import izip

from .. import config
from .. import read_data
from .. import attributes
from .. import utils
from ..reader import TaggerReader

class SRLReader(TaggerReader):
    
    def __init__(self, filename=None, only_boundaries=False, 
                 only_classify=False, only_predicates=False):
        """
        The reader will read sentences from a given file. This file must
        be in the correct format (one token per line, columns indicating
        which tokens are predicates and their argument structure. See
        function read_data.read_plain_srl for more details).
        
        :param filename: a file with CoNLL-like format data. If it is None,
            the reader will be created with no data.
        :param only_boundaries: train to identify only argument boundaries
        :param only_classify: train to classify pre-determined argument
        :param only_predicates: train to identify only predicates
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
        
        super(SRLReader, self).__init__()
        
        if filename is not None:
        
            with open(filename) as f:
                sent_data = read_data.read_conll(f)
            
            sents = []
            tags = []
            preds = []
            for item in sent_data:
                sents.append(item[0])
                tags.append(item[1])
                preds.append(item[2])
            
            self.sentences = zip(sents, tags)
            
            self.predicates = [np.array(x) for x in preds]
            self._clean_text()
            
            # remove this line if working with languages other than Portuguese
            # TODO: paramaterize this behavior
            self.sentences, self.predicates = utils.make_contractions_srl(self.sentences, 
                                                                          self.predicates)
        
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
                new_word = utils.clean_text(token.word, correct=False)
                new_lemma = utils.clean_text(token.lemma, correct=False) 
                token.word = new_word
                token.lemma = new_lemma
                sent[i] = token

                    
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
        """Removes the actual tag names, leaving only IOB or IOBES block delimiters."""
        for _, propositions in self.sentences:
            for tags in propositions:
                for i, tag in enumerate(tags):
                    tags[i] = tag[0]
    
    def _codify_sentences(self):
        """Internal helper function."""
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
    
