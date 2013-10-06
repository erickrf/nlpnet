import sys
from distutils.core import setup
from distutils.extension import Extension

try:
    from Cython.Distutils import build_ext
except ImportError:
    print "You don't seem to have Cython installed. Please get a"
    print "copy from www.cython.org and install it"
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print "You don't seem to have NumPy installed. Please get a"
    print "copy from www.numpy.org and install it"
    sys.exit(1)

setup(
      name = 'nlpnet',
      description = 'Neural networks for NLP tasks.',
      packages = ['nlpnet', 'nlpnet.pos', 'nlpnet.srl'],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("nlpnet.network", 
                               ["nlpnet/network.pyx"],
                               include_dirs=['.', np.get_include()]
                               )
                     ],
      scripts = ['bin/nlpnet-tag.py'],
      license = 'LICENSE.txt',
      version = '1.0.0',
      author = 'Erick Fonseca',
      author_email = 'erickrfonseca@gmail.com'
      )
