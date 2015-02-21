==============
Trained Models
==============

Here, you find trained models ready to be used with :mod:`nlpnet`. Model files can be decompressed anywhere, and when using :mod:`nlpnet`, the path to it must be supplied (using the ``--data`` argument in the ``nlpnet-tag`` script or the ``nlpnet.set_data_dir`` function).

Currently, there are only models for POS and SRL in Portuguese, plus word embeddings not trained for any task. If you have trained :mod:`nlpnet` to perform any task in another language, please enter in contact and we add provide a link to your models.

Word Embeddings
===============

`Word embeddings <data/embeddings-pt.tgz>`_

.. note::
  This is only useful for training new models. If you want to use a pre-trained POS or SRL model, you don't need the embeddings.

These word embeddings can be used to train new :mod:`nlpnet` models (check the :ref:`training` Section for details on how to use them). The archive contains a vocabulary file and an embeddings file. The latter is a NumPy matrix whose *i*-th row corresponds to the vector representation of the *i*-th word in the vocabulary. The embeddings were obtained applying `word2embeddings`_ over a corpus of around 240 million tokens, composed of the Portuguese Wikipedia and news articles.
  
.. _`word2embeddings`: https://bitbucket.org/aboSamoor/word2embeddings

POS
===

`State-of-the-art POS tagger <data/pos-pt.tgz>`_
  
**Performance:** 97.33% token accuracy, 93.66% out-of-vocabulary token accuracy (evaluated on the revised `Mac-Morpho`_ test section)


.. _`Mac-Morpho`: http://nilc.icmc.usp.br/macmorpho
  
SRL
===

`Semantic Role Labeling model <data/srl-pt.tgz>`_

This SRL model doesn't use any feature besides word vectors. You can use it without a parser or a chunker. However, due to the small size of `PropBank-Br`_, its performance is lower than what SENNA obtains for English. 
  
**Performance:** 66.19% precision, 59.78% recall, 62.82 F-1 (evaluated on `PropBank-Br`_ test section)

.. _`PropBank-Br`: http://www.nilc.icmc.usp.br/portlex/index.php/en/projects/propbankbringl
