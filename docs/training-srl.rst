..  _training-srl:

SRL Training
============

Training data
-------------

Training data for SRL is expected to be in a CoNLL-like format. Each line corresponds to a token and blank lines indicate the end of a sentence. Each token should have the following fields, separated by whitespace or tab (non-mandatory fields may have any value)

1. Token number, starting from 1
2. The token itself
3. Lemma (not necessary, but :mod:`nlpnet` may be configured to use lemmas instead of surface forms)
4. Coarse POS tag (not necessary, but may be used as an additional attribute)
5. Fine POS tag / morphological information (not used, just a CoNLL convention)
6. Clause boundaries (not necessary)
7. Chunks (not necessary)
8. Parse tree (not necessary)
9. A dash (``-``) for non-predicates and anything else for predicates
10. (and others) The argument labels. Each column starting from the 10th refers to an predicate, in the order they appear in the sentence. If a token is the only one in an argument, this field must contain ``(ARG-LABEL*)``. If it starts one, it must be ``(ARG-LABEL*``. And if it ends one, it must be ``*)``. Others should have ``*``.

Example with two predicates:

::

    1 Ele                       ele                       pron-pers  - - - - - (A0*)     (A1*)    
    2 só                        só                        adv        - - - - - (AM-ADV*) *        
    3 não                       não                       adv        - - - - - (AM-NEG*) *        
    4 jogava                    jogar                     v-fin      - - - - Y (V*)      *        
    5 porque                    porque                    conj-s     - - - - - (AM-CAU*  *        
    6 não                       não                       adv        - - - - - *         (AM-NEG*)
    7 estava                    estar                     v-fin      - - - - Y *         (V*)     
    8 bem                       bem                       adv        - - - - - *)        (A2*)    
    9 .                         .                         pu         - - - - - *         *    

In this example, most of the fields concerning parse trees are empty. The third and fourth columns (lemmas and POS tags) could also be empty and the data would still be useful for :mod:`nlpnet`. 

Tags are **not** restricted to the Propbank tagset. You can use any tagset, as long as tags don't contain spaces and the training file is encoded in UTF-8.

Invoking the training script
----------------------------

SRL is a relatively complex task when compared to POS tagging, and it reflects on the :mod:`nlpnet` training. It is composed of at least two steps:

* Identifying predicates (and there may be none in a sentence)
* Identifying and labeling semantic arguments for each detected predicate

Collobert and Weston [1]_ performed the second step with SENNA in one go, but it is more commonly split into two: argument boundary identification and classification. :mod:`nlpnet` can do it in both ways. For each subtask, it will train and save a different model, but all of them should be in the same directory.

Training SRL models is done by calling ``nlpnet-train.py srl`` followed by the subtask name, which is either ``pred``, ``id``, ``class`` or ``1step`` (note that ``1step`` refers to argument identification and classification; it still needs predicate detection to be done separately). Mandatory arguments are the training data file (``--gold``) and the directory where the trained models should be saved (``--data``). See subtask specific guidelines below.

Predicate detection
~~~~~~~~~~~~~~~~~~~

Predicate detection is the simplest of the SRL subtasks. It tags each token in a sentence as a predicate or non-predicate. Notice that this is very close to detecting verbs, and may seem as an easier version of POS tagging. However, there are two points to consider:

* Auxiliary verbs are not predicates. For example, the word *is* in a sentence like *"The boy is running"* would be tagged as a verb (VBZ) according to the Penn Treebank annotation style, and the same tag is also used for main verbs. This is also true for other languages.

* The concept of role labeling may also be applied to tasks other than classical verb-based SRL. For example, you could be interested in labeling noun arguments.

The predicate detection network is similar to the POS tagging network. It only examines a window of tokens at a time, and doesn't need convolution. You call it with:

.. code-block:: bash

    $ nlpnet-train.py srl pred --gold /path/to/training-data.txt --data srl-model/

As in :ref:`training-pos`, you can load existing word embeddings:

.. code-block:: bash

    $ nlpnet-train.py srl pred --gold /path/to/training-data.txt --data srl-model/ --load_features

You can continue the training of a previously saved model using ``--load_network``. The predicate detector runs by default a single epoch with the learning rates set to 0.01. It might be a good idea to use ``--decay`` in order to lower learning rates during training.

Argument identification and classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Argument identification consists of tagging all tokens of a sentence with IOBES tags (In, Out, Begin, End, Single) relative to a given predicate. The neural network for this task performs a temporal convolution.

By default, it uses a window of 3 tokens, 150 neurons in the convolution layer and 150 in the hidden layer after that. It is trained for 15 epochs with learning rates set to 0.001. You may change the number of convolution neurons with ``-c`` and hidden ones with ``-n``. For example:

.. code-block:: bash

    $ nlpnet-train.py srl id --gold /path/to/training-data.txt --data srl-model/ -c 200 -n 250

In experiments with Portuguese, a little improvement was achieved after loading the initial model and training it further with learning rates at 0.0001:

.. code-block:: bash

    $ nlpnet-train.py srl id --gold /path/to/training-data.txt --data srl-model/ --load_network -e 10 -l 0.0001 --lf 0.0001 --lt 0.0001

The argument classification model receives the output of the previous one and tags each argument block with the right tag. By default, it doesn't have a hidden layer after the convolution (same as ``-n 0``). In these cases, a non-linear function is applied directly to the output of the convolution. Still, you can force :mod:`nlpnet` to use an additional layer:

.. code-block:: bash

    $ nlpnet-train.py srl class --gold /path/to/training-data.txt --data srl-model/ -n 100

In experiments with Portuguese SRL, it was useful to start training the argument classification model with 3 epochs with the learning rates set to 0.01 and then 10 to 15 with 0.001. The training script defaults to use the initial values, and you can then load the trained model and supply lower rates:

.. code-block:: bash

    $ nlpnet-train.py srl class --gold /path/to/training-data.txt --data srl-model/ --load_network -e 15 -l 0.001 --lf 0.001 --lt 0.001
    
Finally, the one-step model is an alternative to the combination of the two above. In experiments with Portuguese, it yielded slightly worse results [2]_, but you may still use it with :mod:`nlpnet`. It tags all tokens in the sentence with a combination of IOB (In, Out, Begin) and the argument labels.

It defaults to use 200 convolution neurons and 150 in the following hidden layer, and 15 epochs with learning rates set to 0.001. As with argument identification, it was found useful to perform a few more epochs with the rate set to 0.0001.

A few more options
~~~~~~~~~~~~~~~~~~

All of the three models for argument identification and classification can also use a few more parameters. One of them is the maximum distance with its own feature vector. When it performs convolution, the network computes the distance from each token to the target (the one that is being tagged) and to the predicate. By default, each distance up to 10 has a vector. It means that distances of 11 and higher are treated as being the same. You can change this value with ``-max_dist``.

The size of the distance feature vectors can be set by ``--target_features`` and ``--pred_features``. Both default to 5.

It is often useful to activate verbose mode. Add ``-v`` for more detailed output.

Capitalization may be added as an additional attribute with ``--caps`` (optionally followed by the size of the feature vectors). However, it doesn't seem to be beneficial to SRL.

References
----------

.. [1] Collobert, R. and Weston, J. *A Unified Architecture for Natural Language Processing: Deep Neural Networks with Multitask Learning*. In International Conference on Machine Learning, ICML, 2008.

.. [2] Fonseca, E. R. and Rosa, J.L.G. *A Two-Step Convolutional Neural Network Approach for Semantic
  Role Labeling*. Proceedings of the 2013 International Joint Conference on Neural Networks, 2013.
  p. 2955-2961
