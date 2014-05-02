==============
Trained Models
==============

Here, you find trained models ready to be used with :mod:`nlpnet`. Model files can be decompressed anywhere, and when using :mod:`nlpnet`, the path to it must be supplied (using the ``--data`` argument in the ``nlpnet-tag`` script or the ``nlpnet.set_data_dir`` function).

Currently, there are only models for POS and SRL in Portuguese. If you have trained :mod:`nlpnet` to perform any task in another language, please enter in contact and we add provide a link to your models.

POS
===

`State-of-the-art POS tagger <http://nilc.icmc.usp.br/nlpnet/nlpnet-pos.zip>`_. 
  **Note:** it includes capitalization features. When dealing with text that doesn't use capital letters consistently, like user content from the Web, it may be better to use the non-caps model.

`Non-caps POS tagger <http://nilc.icmc.usp.br/nlpnet/nlpnet-pos-nocaps.zip>`_. 
  This model is usually better to tag text where capital letters are not used consistently. It performs a little worse than the above model, but its accuracy is still state-of-the-art as long as non-caps models are concerned.
  
SRL
===

*Coming soon.*
