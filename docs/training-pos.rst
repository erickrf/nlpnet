..  _training-pos:

POS Training
============

Training data
-------------

First of all, the training data (aka gold standard data). For POS tagging, :mod:`nlpnet` can read data in two formats:

* Files with one sentence per line, having tokens and tags concatenated by an underscore character:

  ::

    This_DT is_VBZ an_DT example_NN (...)
  
* CoNLL format, with one token per line, and lines formed of at least for columns (separated by tabs os whitespace): token number, token itself, lemma (not needed) and the actual tag. Sentences are separated by a blank line.
  
  ::
  
    1 This _ DT
    2 is _ VBZ
    3 an _ DT
    4 example _ NN
    
    1 It _ PRP
    2 is _ VBZ
    3 easy _ JJ
  
:mod:`nlpnet` automatically understands the format used (actually, it first tries to read as the first format and, if it can't, it tries the second one). Tags can be lower or uppercase, and also have accents or other special characters. If your training file is not pure ASCII text, it must be encoded in UTF-8. 

Invoking the training script
----------------------------

In order to train a POS tagger, you have to supply ``nlpnet-train.py`` with at least the training data file (``--gold``):

.. code-block:: bash

    $ nlpnet-train.py pos --gold /path/to/training-data.txt

If you are using previously trained word representations, they must already be in the directory, and you must include ``--load_features`` (if you use ``--data`` to set a different directory for your model, the features file must be there instead):

.. code-block:: bash

    $ nlpnet-train.py pos --gold /path/to/training-data.txt --load_features

If you don't tell :mod:`nlpnet` to load existing embeddings, it will create random vectors for each word type appearing at least twice in your training data. However, if there is file named ``vocabulary.txt`` in the data directory, only vectors for those word types will be created. Use it if you to control which words get their own vectors.

.. note::

  If you load pre-trained word embeddings to initialize your model, the embeddings file is *NOT* changed. Its contents will be copied to the new network.

You can also use additional attributes for POS tagging: capitalization, suffixes and prefixes. These are toggled by ``--caps``, ``--suffix`` and ``--prefix``. Each may be optionally followed by the desired size of the feature vector associated with that attribute (with a default of 2). You may also supply ``--suffix_size`` and ``--prefix_size`` to inform the maximum suffix/prefix size that should be examined in each word (defaults to 5). So, you could use:

.. code-block:: bash
    
    $ nlpnet-train.py pos --gold /path/to/training-data.txt --load_features --suffix --caps 10
    
This would mean that each word is represented as a concatenation of:

* its own feature vector (usually with 50 dimensions)
* a vector for each suffix from sizes 1 to 5 (default value of the unused ``--suffix_size``), each with 2 dimensions, totalling 10
* a vector with 10 dimensions for capitalization

Which would make each token have a resulting 70 dimension vector.

:mod:`nlpnet` saves and loads suffixes and prefixes to files ``suffixes.txt`` and ``prefixes.txt`` in its data directory. By default, affixes are selected if they occur in at least 5 word types with length greater than the affix itself. If you want to customize which suffixes/prefixes are used, just provide custom files.

With this minimal setup, :mod:`nlpnet` will use default parameters for POS tagging, which yielded good results on experiments but may or may not be the best for your needs. Here are some guidelines to help tweak the network.

The model architecture for POS tagging is relatively simple. It consists of a multilayer perceptron neural network, a tag transition score matrix, and word embeddings. The input window default size is 5, and this seems a very good number in experiments with Portuguese and English.

The number of hidden neurons (``-n``) defaults to 100. It is difficult to tell how many are ideal, but this number yielded state-of-the-art performance in a Portuguese corpus with 26 tags. SENNA, trained on the Penn Treebank with 45, uses 300. As a rule of thumb, the more tags you have, the more neurons you need.

:mod:`nlpnet` allows the learning rate of network connections (``-l``), transition scores (``--lt``) and feature values (``--lf``) to be set separately. However, I found that the best results were obtained with all three being equal. 

The learning rates may be decreased with each epoch using the decay option (``--decay``). The best results obtained in Portuguese initialized all rates to 0.01 and used a decay of 1, which means that in each epoch *i*, the learning rate was equal to :math:`0.01 / i`.

The number of epochs (``-e``) is set to 15. 

If the network seems to overfit the data, there is the "desired accuracy" option (``-a``), which sets a value between 0 and 1. When the network achieves this accuracy, training ends. The default value of 0 means that this option is ignored.

If you have a trained model and want to continue training it (maybe with lower learning rates), you can use the following:

.. code-block:: bash

    $ nlpnet-train.py pos --gold /path/to/training-data.txt --data pos-model/ --load_network

You don't need to provide extra attribute options such as ``--caps`` if your model originally used it. This information is saved with the network.
    
Calling ``nlpnet-train.py pos -h`` shows a description of all command line arguments.