===============================================================
``nlpnet`` --- Natural Language Processing with neural networks
===============================================================

``nlpnet`` is a Python library for Natural Language Processing tasks based on neural networks. 
Currently, it performs part-of-speech tagging and semantic role labeling. Most of the 
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

Basic usage
-----------

You can use `nlpnet` as a library in Python code. In order to use the existing models, you will need to download them from http://nilc.icmc.usp.br/nilc/download/nlpnet-data.zip and uncompress it somewhere. The library can be used as follows:

    >>> import nlpnet
    >>> nlpnet.set_data_dir('/path/to/nlpnet-data/')
    >>> tagger = nlpnet.POSTagger()
    >>> tagger.tag('O rato roeu a roupa do rei de Roma.')
        [[(u'O', u'ART'), (u'rato', u'N'), (u'roeu', u'V'), (u'a', u'ART'), (u'roupa', u'N'), (u'do', u'PREP+ART'), (u'rei', u'N'), (u'de', u'PREP'), (u'Roma', u'NPROP'), (u'.', 'PU')]]

`nlpnet` also provides scripts for tagging text, training new models and testing them. They are copied to the `scripts` subdirectory of your Python installation. You can call them from command line and give some text input.

.. code-block:: bash

    $ nlpnet-tag.py pos /path/to/nlpnet-data/
    O rato roeu a roupa do rei de Roma.
    O_ART rato_N roeu_V a_ART roupa_N do_PREP+ART rei_N de_PREP Roma_NPROP ._PU

Or with semantic role labeling:

.. code-block:: bash

    $ nlpnet-tag.py srl /path/to/nlpnet-data/
    O rato roeu a roupa do rei de Roma.
    O rato roeu a roupa do rei de Roma .
    roeu
        A1: a roupa do rei de Roma
        A0: O rato
        V: roeu

The first line was typed by the user, and the second one is the result of tokenization.

To learn more about training and testing new models, and other functionalities, refer to the documentation at http://nilc.icmc.usp.br/nilc/tools/nlpnet
