============
Introduction
============

This documents covers the basics for installing and using :mod:`nlpnet`. 

Installation
------------

:mod:`nlpnet` can be downloaded from the Python package index at https://pypi.python.org/pypi/nlpnet/ or installed with

.. code-block:: bash

    pip install nlpnet

See the `Dependencies`_ section below for additional installation requirements.
    
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

Brief explanation
-----------------

Here is a brief exaplanation about how stuff works in the internals of :mod:`nlpnet` (*you don't need
to know it to use this library*).
For a more detailed view, refer to the articles in the index page or about the SENNA system.

Two types of neural networks are available: the common MLP (multilayer perceptron) and the convolutional one. 
The former was used to train a POS model, and the latter an SRL model. Basically, the common MLP examines
word windows, outputs a score for assigning each tag to each word, and then determines 
the tags using the Viterbi algorithm (which is essentially picking the best combination from network
scores and tag transition scores).

During training, adjustments are made to the network connections, word representations and 
the tag transition scores. Their learning rates may be set separately, although the best
results seem to arise when all three have the same value.

The convolutional network is a little more complicated. In order to output a score for each 
word, it examines the whole sentence. It does so by picking a word window at a time and forwarding it to 
a convolution layer. This layer stores in each of its neurons the biggest value found so far.
After all words have been examined, the convolution layer forwards its output like a usual MLP network.
Then, it works like the previous model: the network outputs scores for each word/tag combination,
and a Viterbi search is performed.

In the convolution layer, the values found by each neuron may come from different words, i.e., each neuron stores
its maximum independently from the others. This is particularly complex during training, because 
neurons must backpropagate their error only to the word window that yielded their stored value.

All the details concerning the neural networks are hidden from the user when calling the tagger methods or 
the ``nlpnet-tag`` standalone script. However, they are available to play with in the :ref:`network` module.

Basic usage
-----------

:mod:`nlpnet` can be used both as a Python library or by its standalone scripts. The basic library API is explained below.
See also :ref:`scripts`.

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

