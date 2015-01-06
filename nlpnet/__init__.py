# import to provide easier access for nlpnet users
from .config import set_data_dir
from . import taggers
from . import utils

from .taggers import POSTagger, SRLTagger
from .utils import tokenize
