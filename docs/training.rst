========
Training
========

New models for POS and SRL can be trained using the ``nlpnet-train.py`` script. It is copied to the *scripts* subdirectory of your Python installation, which can be included in the system PATH variable. 

Here is explained how to use it to train models for POS tagging and Semantic Role Labeling. All command line options mentioned below (such as ``-n``, ``-w``, ``--load_network``, etc.) should be given to the ``nlpnet-train.py`` algorithm.

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

The vocabulary must have one word type per line, encoded in UTF-8. The vocabulary is also treated as case insensitive, so, if you have an entry for "Apple" and another for "apple", one of them will be ignored (naturally, :mod:`nlpnet` *can* check capital letters when tagging text, but it observes the presence of upper case letters and the vocabulary separately). Additionally, all digits are internally replaced by 9's, so there's no point in using digits 0-8 in the vocabulary. It must be saved to a file called ``vocabulary.txt``.

The embeddings must be stored in a 2-dim :mod:`numpy` array, such that the *i*-th row corresponds to the *i*-th word in the vocabulary. This matrix should be saved using the default :mod:`numpy` save command. The file name depends on the task you want to use the embeddings for:

* POS tagging: ``types-features-pos.npy``
* SRL predicate detection (check the SRL section for details about the different SRL modes): ``types-features-preds.npy``
* SRL in one step: ``types-features-1step.npy``
* SRL argument delimitation: ``types-features-id.npy``
* SRL argument classification: ``types-features-class.npy``

Importing embeddings with the nlpnet-load-embeddings script
-----------------------------------------------------------

The ``nlpnet-load-embeddings.py`` script can read input files in different formats and import them to be used by :mod:`nlpnet`. Currently, it deals with embeddings in the following formats:

1. Plain text (also those of SENNA, which include ``PADDING`` and ``UNKNOWN`` in the vocabulary). These embeddings are stored with one vector per line.
2. word2embeddings_
3. gensim_

You must also provide a vocabulary file (except for gensim embeddings, which saves vocabulary and their vectors together). The same recommendations mentioned in `Importing embeddings manually`_ apply for this file: UTF-8 encoding, everything is converted to lowercase and digits are replaced by 9's.

Here's how to call ``nlpnet-load-embeddings.py`` from the command line:

.. code-block:: bash

    $ nlpnet-load-embeddings.py [FORMAT] [EMBEDDINGS_FILE] -v [VOCABULARY_FILE] -o [OUTPUT_DIRECTORY]
    
``FORMAT`` is one of ``senna``, ``plain``, ``word2embeddings`` and ``gensim``. The vocabulary isn't used with gensim vectors and the output defaults to the current directory.

POS Training
============

First of all, the training data (aka gold standard data). For POS tagging, :mod:`nlpnet` expects files with one sentence per line, having tokens and tags concatenated by an underscore character:

::

    Token_TAG token_TAG token_TAG (...)

Tags can be lower or uppercase, and also have accents or other special characters. If your training file is not pure ASCII text, it must be encoded in UTF-8. 

In order to train a POS tagger, you have to supply ``nlpnet-train.py`` with at least the training data file (``--gold``) and the directory where the trained models should be saved (``--data``): 

.. code-block:: bash

    $ nlpnet-train.py pos --gold /path/to/training-data.txt --data pos-model/

If you are using previously trained word representations, they must already be in the data directory, and you must include ``--load_features``:

.. code-block:: bash

    $ nlpnet-train.py pos --gold /path/to/training-data.txt --data pos-model/ --load_features

If you don't tell :mod:`nlpnet` to load existing embeddings, it will create random vectors for each word type appearing at least twice in your training data. However, if there is file named ``vocabulary.txt`` in the data directory, only vectors for those word types will be created. Use it if you to control which words get their own vectors.

You can also use additional attributes for POS tagging: capitalization, suffixes and prefixes. These are toggled by ``--caps``, ``--suffix`` and ``--prefix``. Each may be optionally followed by the desired size of the feature vector associated with that attribute (with a default of 2). You may also supply ``--suffix_size`` and ``--prefix_size`` to inform the maximum suffix/prefix size that should be examined in each word (defaults to 5). So, you could use:

.. code-block:: bash
    
    $ nlpnet-train.py pos --gold /path/to/training-data.txt --data pos-model/ --load_features --suffix --caps 10
    
This would mean that each word is represented as a concatenation of:

* its own feature vector (usually with 50 dimensions)
* a vector for each suffix from sizes 1 to 5 (default value of the unused ``--suffix_size``), each with 2 dimensions, totalling 10
* a vector with 10 dimensions for capitalization

Which would make each token have a resulting 70 dimension vector.

:mod:`nlpnet` saves and loads suffixes and prefixes to files ``suffixes.txt`` and ``prefixes.txt`` in its data directory. By default, affixes are selected if they occur in at least 5 word types with length greater than the affix itself. If you want to customize which suffixes/prefixes are used, just provide custom files.

With this minimal setup, :mod:`nlpnet` will use default parameters for POS tagging, which yielded good results on experiments but may or may not be the best for your needs. Here are some guidelines to help tweak the network.

The model architecture for POS tagging is relatively simple. It consists of a multilayer perceptron neural network, a tag transition score matrix, and word embeddings. The input window default size is 5, and this seems a very good number in experiments with Portuguese and English.

The number of hidden neurons (``-n``) defaults to 100. It is difficult to tell how many are ideal, but this number yielded state-of-the-art performance in a Portuguese corpus with 26 tokens. SENNA, trained on the Penn Treebank with 45, uses 300. As a rule of thumb, the more tags you have, the more neurons you need.

:mod:`nlpnet` allows the learning rate of network connections (``-l``), transition scores (``--lt``) and feature values (``--lf``) to be set separately. However, I found that the best results were obtained with all three being equal, and they all default to 0.001. The number of epochs (``-e``) is set to 15. One possibility is to train the network for a few epochs with a given learning rate and then train it further with lower rates.

If the network seems to overfit the data, there is the "desired accuracy" option (``-a``), which sets a value between 0 and 1. When the network achieves this accuracy, training ends. The default value of 0 means that this option is ignored.

If you have a trained model and want to continue training it (maybe with lower learning rates), you can use the following:

.. code-block:: bash

    $ nlpnet-train.py pos --gold /path/to/training-data.txt --data pos-model/ --load_features --load_network

You don't need to provide extra attribute options such as ``--caps`` if your model originally used it. This information is saved with the network.
    
Calling ``nlpnet-train.py pos -h`` shows a description of all command line arguments.