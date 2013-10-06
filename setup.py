import sys
from distutils.core import setup
from distutils.extension import Extension

try:
    import numpy as np
except ImportError:
    print "You don't seem to have NumPy installed. Please get a"
    print "copy from www.numpy.org and install it"
    sys.exit(1)

setup(
      name = 'nlpnet',
      description = 'Neural networks for NLP tasks',
      packages = ['nlpnet', 'nlpnet.pos', 'nlpnet.srl'],
      ext_modules = [Extension("nlpnet.network", 
                               ["nlpnet/network.c"],
                               include_dirs=['.', np.get_include()]
                               )
                     ],
      scripts = ['bin/nlpnet-tag.py',
                 'bin/nlpnet-train.py',
                 'bin/nlpnet-test.py'],
      license = 'LICENSE.txt',
      version = '1.0.0',
      author = 'Erick Fonseca',
      author_email = 'erickrfonseca@gmail.com'
      )
