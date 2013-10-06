===============================================================
``nlpnet`` --- Natural Language Processing with neural networks
===============================================================

``nlpnet`` is a Python library for Natural Language Processing tasks based on neural networks. 
Currently, it performs only part-of-speech tagging and semantic role labeling. Most of the 
architecture is language independent, but some functions were specially tailored for working
with Portuguese.

This system was inspired by SENNA_, but has some conceptual and practical differences. 
If you use ``nlpnet``, please cite one or both of the articles below, according to your needs (POS or
SRL):

.. _SENNA: http://ronan.collobert.com/senna/

* Fonseca, E. R. and Rosa, J.L.G. *A Two-Step Convolutional Neural Network Approach for Semantic
  Role Labeling*. Proceedings of the 2013 International Joint Conference on Neural Networks, 2013.
  p. 2955-2961.

* Fonseca, E. R. and Rosa, J.L.G. *Mac-Morpho Revisited: Towards Robust Part-of-Speech Tagging*. 
  Proceedings of the 9th Brazilian Symposium in Information and Human Language Technology, 2013. p.  
  98-107 [`PDF <http://aclweb.org/anthology//W/W13/W13-4811.pdf>`_]

Dependencies
------------

``nlpnet`` requires NLTK_ and numpy_. Additionally, it needs to download some data from NLTK. After installing it, call

    >>> nltk.download()

go to the `Models` tab and select the Punkt tokenizer. It is used in order to split the text into sentences.

Cython_ is used to generate C extensions and run faster. You probably won't need it, since the generated ``.c`` file is already provided with ``nlpnet``, but you will need a C compiler. On Linux and Mac systems this shouldn't be a problem, but may be on Windows, because ``setuptools`` requires the Microsoft C Compiler by default. If you don't have it already, it is usually easier to install MinGW_ instead and follow the instructions `here <http://docs.cython.org/src/tutorial/appendix.html>`_.

.. _NLTK: http://www.nltk.org
.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org
.. _MinGW: http://www.mingw.org

Contents:

.. toctree::
   :maxdepth: 2



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

