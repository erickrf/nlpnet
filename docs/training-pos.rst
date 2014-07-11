..  _training-pos:

POS Training
============

Training data
-------------

First of all, the training data (aka gold standard data). For POS tagging, :mod:`nlpnet` expects files with one sentence per line, having tokens and tags concatenated by an underscore character:

::

    Token_TAG token_TAG token_TAG (...)

Tags can be lower or uppercase, and also have accents or other special characters. If your training file is not pure ASCII text, it must be encoded in UTF-8. 

Invoking the training script
----------------------------

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