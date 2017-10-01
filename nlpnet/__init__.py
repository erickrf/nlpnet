# import to provide easier access for nlpnet users
from nlpnet.config import set_data_dir
import nlpnet.taggers
import nlpnet.utils

from nlpnet.taggers import POSTagger, SRLTagger, DependencyParser
from nlpnet.utils import tokenize

__version__ = '1.2.2'
