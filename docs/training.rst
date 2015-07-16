.. _training:

========
Training
========

New models for POS and SRL can be trained using the ``nlpnet-train.py`` script. It is copied to the *scripts* subdirectory of your Python installation, which can be included in the system PATH variable. 

Here is explained how to use it to train models for POS tagging and Semantic Role Labeling. All command line options mentioned below (such as ``-n``, ``-w``, ``--load_network``, etc.) should be given to the ``nlpnet-train.py`` algorithm.

.. _embeddings:

Importing Word Representations
==============================

You probably want to use word representations previously trained on a large corpus in order to train a POS tagger or SRL network. Initializing a model with pre-trained embeddings is one of the main advantages of this architecture. If you don't want to pre-train word embeddings, just skip this section. You can have :mod:`nlpnet` generate random vectors for words it finds in training files.

:mod:`nlpnet` doesn't provide any functionality for training such models, but there are some good implementations available out there you can use (and then import them to be used by :mod:`nlpnet`):

* word2embeddings_ is an efficient implementation of the neural language model introduced by Ronan Collobert and Jason Weston.
* word2vec_ implements the skip-gram model. In my experiments, it yielded a slightly worse performance than the neural model, but it is much faster to generate.
* gensim_ is primarily targeted at topic analysis, but also includes an implementation of skip-gram above with a Python interface.
* `Semantic Vectors`_ implements distributional semantics techniques like LSA, HAL and Random Indexing. They are not the best choice for training a deep learning based neural network, but maybe you want to try something different.

.. _word2embeddings: https://bitbucket.org/aboSamoor/word2embeddings
.. _word2vec: https://code.google.com/p/word2vec/   
.. _gensim: http://radimrehurek.com/gensim/index.html
.. _`Semantic Vectors`: https://code.google.com/p/semanticvectors/

*(if you want to suggest any other relevant software for word embeddings, feel free to contact me)*

Once you have your embeddings properly trained, you can import them to :mod:`nlpnet`. You can do it manually or use the provided ``nlpnet-load-embeddings.py`` script.

Importing embeddings manually
-----------------------------

You can save your word embeddings directly in the format used by :mod:`nlpnet`. You will need to create two files: the vocabulary and the actual embeddings. 

The vocabulary must have one word type per line, encoded in UTF-8. The vocabulary is also treated as case insensitive, so, if you have an entry for "Apple" and another for "apple", one of them will be ignored (naturally, :mod:`nlpnet` *can* check capital letters when tagging text, but it observes the presence of upper case as an independent feature). Additionally, all digits are internally replaced by 9's, so there's no point in using digits 0-8 in the vocabulary. It must be saved to a file called ``vocabulary.txt``.

The embeddings must be stored in a 2-dim :mod:`numpy` array, such that the *i*-th row corresponds to the *i*-th word in the vocabulary. This matrix should be saved using the default :mod:`numpy` save command. The file name must be ``types-features.npy``.

Importing embeddings with the nlpnet-load-embeddings script
-----------------------------------------------------------

The ``nlpnet-load-embeddings.py`` script can read input files in different formats and import them to be used by :mod:`nlpnet`. Currently, it deals with embeddings in the following formats:

1. Plain text (also those of SENNA, which include ``PADDING`` and ``UNKNOWN`` in the vocabulary). These embeddings are stored with one vector per line.
2. word2embeddings_
3. gensim_
4. polyglot_
5. Single file containing vocabulary and embeddings. Everything should be separated by whitespaces.

.. _`polyglot`: https://sites.google.com/site/rmyeid/projects/polyglot

You must also provide a vocabulary file (except for gensim and single file embeddings, which saves vocabulary and their vectors together). The same recommendations mentioned in `Importing embeddings manually`_ apply for this file: UTF-8 encoding, everything is converted to lowercase and digits are replaced by 9's.

.. note::
  The Polyglot project provides different embeddings for words with varying case and with digits. In order to make them compatible with :mod:`nlpnet`, the vectors for all case variations of a word are averaged. This leads to some unavoidable knowledge loss.

Here's how to call ``nlpnet-load-embeddings.py`` from the command line:

.. code-block:: bash

    $ nlpnet-load-embeddings.py [FORMAT] [EMBEDDINGS_FILE] -v [VOCABULARY_FILE] -o [OUTPUT_DIRECTORY]
    
``FORMAT`` is one of ``senna``, ``plain``, ``word2embeddings``, ``gensim``, ``polyglot`` and ``single``. The vocabulary isn't used with gensim vectors and the output defaults to the current directory.

Task specific training
======================

.. toctree::
   :maxdepth: 1
   
   training-pos
   training-srl
