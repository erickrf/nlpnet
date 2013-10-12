============
Introduction
============

This documents covers the basics for installing and using :mod:`nlpnet`. 

Installation
------------

:mod:`nlpnet` can be downloaded from the Python package index at https://pypi.python.org/pypi/nlpnet/ or installed with

.. code-block:: bash

    pip install nlpnet

**Important:** in order to use the trained models for Portuguese NLP, you will need to download the data from http://nilc.icmc.usp.br/nilc/download/nlpnet-data.zip and unzip it into some directory.

Dependencies
~~~~~~~~~~~~

:mod:`nlpnet` requires NLTK_ and numpy_. Additionally, it needs to download some data from NLTK. After installing it, call

    >>> nltk.download()

go to the `Models` tab and select the Punkt tokenizer. It is used in order to split the text into sentences.

Cython_ is used to generate C extensions and run faster. You probably won't need it, since the generated ``.c`` file is already provided with :mod:`nlpnet`, but you will need a C compiler. On Linux and Mac systems this shouldn't be a problem, but may be on Windows, because  setuptools_ requires the Microsoft C Compiler by default. If you don't have it already, it is usually easier to install MinGW_ instead and follow the instructions `here <http://docs.cython.org/src/tutorial/appendix.html>`_.

.. _NLTK: http://www.nltk.org
.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org
.. _MinGW: http://www.mingw.org
.. _setuptools: http://pythonhosted.org/setuptools/

Basic usage
-----------

:mod:`nlpnet` can be used both as a Python library or by its standalone scripts. Both usages are explained below.

Library usage
~~~~~~~~~~~~~

You can use :mod:`nlpnet` as a library in Python code as follows:

.. code-block:: python

    >>> import nlpnet
    >>> nlpnet.set_data_dir('/path/to/nlpnet-data/')
    >>> tagger = nlpnet.POSTagger()
    >>> tagger.tag('O rato roeu a roupa do rei de Roma.')
    [[(u'O', u'ART'), (u'rato', u'N'), (u'roeu', u'V'), (u'a', u'ART'), (u'roupa', u'N'), (u'do', u'PREP+ART'), (u'rei', u'N'), (u'de', u'PREP'), (u'Roma', u'NPROP'), (u'.', 'PU')]]

In the example above, the call to ``set_data_dir`` indicates where the data directory is located. This location must be given whenever :mod:`nlpnet` is imported. 

Calling a tagger is pretty straightforward. The two provided taggers are ``POSTagger`` and ``SRLTagger``, both having a method ``tag`` which receives strings with text to be tagged. The tagger splits the text into sentences and then tokenizes each one (hence the return of the POSTagger is a list of lists).

The output of the SRLTagger is slightly more complicated:

    >>> tagger = nlpnet.SRLTagger()
    >>> tagger.tag(u'O rato roeu a roupa do rei de Roma.')
    [<nlpnet.taggers.SRLAnnotatedSentence at 0x84020f0>]

Instead of a list of tuples, sentences are represented by instances of ``SRLAnnotatedSentence``. This class serves basically as a data holder, and has two attributes:

    >>> sent = tagger.tag(u'O rato roeu a roupa do rei de Roma.')[0]
    >>> sent.tokens
    [u'O', u'rato', u'roeu', u'a', u'roupa', u'do', u'rei', u'de', u'Roma', u'.']
    >>> sent.arg_structures
    [(u'roeu',
      {u'A0': [u'O', u'rato'],
       u'A1': [u'a', u'roupa', u'do', u'rei', u'de', u'Roma'],
       u'V': [u'roeu']})]

The ``arg_structures`` is a list containing all predicate-argument structures in the sentence. The only one in this example is for the verb `roeu`. It is represented by a tuple with the predicate and a dictionary mapping semantic role labels to the tokens that constitute the argument.

Note that the verb appears as the first member of the tuple and also as the content of label 'V' (which stands for verb). This is because some predicates are multiwords. In these cases, the "main" predicate word (usually the verb itself) appears in ``arg_structures[0]``, and all the words appear under the key 'V'.

Standalone scripts
~~~~~~~~~~~~~~~~~~

:mod:`nlpnet` also provides scripts for tagging text, training new models and testing them. They are copied to the `scripts` subdirectory of your Python installation, which can be included in the system PATH variable. You can call them from command line and give some text input.

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
