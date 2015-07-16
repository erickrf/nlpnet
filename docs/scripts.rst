.. _scripts:

==================
Standalone Scripts
==================

:mod:`nlpnet` includes standalone scripts that may be called from a command line. They are 
copied to the `scripts` subdirectory of your Python installation, which can be included 
in the system PATH variable. There are three such scripts:

**nlpnet-train**
  Script to train a new model or further train an existing one. See :ref:`training` for detailed information on how to use it.

**nlpnet-load-embeddings**
  Script to load word embeddings trained externally. It accepts different formats. See :ref:`embeddings` for detailed information on how to use it.
  
**nlpnet-test**
  Script to measure the performance of a model against a gold data set.

**nlpnet-tag**
  Script to call a model and tag some given text.

Each of them is explained below.

.. contents::  
  :local:  
  :depth: 1  


nlpnet-tag
==========

This is the simplest :mod:`nlpnet` script. It simply runs the system for a given text input. 
It should be called with the following syntax:

.. code-block:: bash

    $ nlpnet-tag.py TASK

Where ``TASK`` is either ``pos`` or ``srl``. It has also the following command line options:

-v  Verbose mode.
-t  Disables built-in tokenizer. Tokens are assumed to be separated by whitespace and one sentence per line.
--lang  Sets the tokenkizer language (ignored if ``-t`` is used). Currently, it only accepts ``pt`` and ``en``. 
--no-repeat  Forces the classification step to avoid repeated argument labels (SRL only).
--data  The directory with the trained models (defaults to the current one).

For example:

.. code-block:: bash

    $ nlpnet-tag.py pos --data /path/to/nlpnet-data/ --lang pt
    O rato roeu a roupa do rei de Roma.
    O_ART rato_N roeu_V a_ART roupa_N do_PREP+ART rei_N de_PREP Roma_NPROP ._PU

Or with semantic role labeling:

.. code-block:: bash

    $ nlpnet-tag.py srl --data /path/to/nlpnet-data/ --lang pt
    O rato roeu a roupa do rei de Roma.
    O rato roeu a roupa do rei de Roma .
    roeu
        A1: a roupa do rei de Roma
        A0: O rato
        V: roeu

The first line was typed by the user, and the second one is the result of tokenization.


nlpnet-test
===========

This script is much simpler. It evaluates the system performance against a gold standard. 

General options
---------------

The arguments below are valid for both tasks.

--task TASK  Task for which the network should be used. Either ``pos`` or ``srl``.
-v  Verbose mode
--gold FILE  File with gold standard data
--data DIRECTORY  Directory with trained models

POS
---

--oov FILE  Analyze performance on the words described in the given file.

The ``--oov`` option requires a UTF-8 file containing one word per line. Actually, this option
is not exclusive for OOV (out-of-vocabulary) words, but rather any word list you
want to evaluate.

SRL
---

SRL evaluation is performed in different ways, depending on whether it is aimed at
argument identification, classification, predicate detection or all of them.
In the future, there may be a more standardized version for this test.

--id  Evaluate only argument identification (SRL only). The script will output the score.
--class  Evaluate only argument classification (SRL only). The script will output the score.
--preds  Evaluate only predicate identification (SRL only). The script will output the score.
--2steps  Execute SRL with two separate steps. The script will output the results in CoNLL format.
--no-repeat  Forces the classification step to avoid repeated argument labels (2 step SRL only)
--auto-pred  Determines SRL predicates automatically. Only used when evaluating the full process (identification + classification)

The CoNLL output can be evaluated against a gold file using the official SRL eval script (see http://www.lsi.upc.edu/~srlconll/soft.html).


