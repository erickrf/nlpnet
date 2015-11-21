import sys
import warnings
from distutils.core import setup
from distutils.extension import Extension

try:
    import numpy as np
except ImportError:
    warnings.warn(
        "You don't seem to have NumPy installed. "
        "Please get a copy from www.numpy.org and install it."
    )
    sys.exit(1)

def readme():
    with open('README.rst') as f:
        text = f.read()
    return text

setup(
    name = 'nlpnet',
    description = 'Neural networks for NLP tasks',
    packages = ['nlpnet', 'nlpnet.pos', 'nlpnet.srl', 'nlpnet.parse'],
    ext_modules = [
        Extension(
            "nlpnet.network", 
            ["nlpnet/network.c"],
            include_dirs=['.',
            np.get_include()]
        )
    ],
    scripts = [
        'bin/nlpnet-tag.py',
        'bin/nlpnet-train.py',
        'bin/nlpnet-test.py',
        'bin/nlpnet-load-embeddings.py'
    ],
    license = 'MIT',
    version = '1.2.1',
    author = 'Erick Fonseca',
    author_email = 'erickrfonseca@gmail.com',
    url = 'http://nilc.icmc.usp.br/nlpnet',
    long_description = readme()
)
