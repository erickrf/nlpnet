===============================================================
``nlpnet for Korean`` --- Korean Natural Language Processing with neural networks
===============================================================

``nlpnet for Korean``은 ``neural network``를 이용한 한국어 자연어 처리용 파이썬 라이브러리입니다.
현재, 기존의 nlpnet_ 프로젝트를 가져와 POS만 한글에 맞게 수정한 상태입니다.
앞으로 nlpnet_ 의 SRL과 DEPENDENCY Parsing부분도 수정해야 할 것입니다.
기존의 nlpnet_ 프로젝트의 설명에 따르면 거의 대부분의 구조가 언어에 종속적이지 않으나, 몇몇 함수들은 포르투갈어에 맞게 설계되었다고 합니다.
또한 SENNA_ 에 영감을 받아 시작하였다고도 서술하고 있습니다.

이 페이지에는 일부 오역이 포함되어 있을 수 있습니다. 원문은 nlpnet_ 에서 확인해 주세요.

.. _nlpnet: https://github.com/erickrf/nlpnet/
.. _SENNA: http://ronan.collobert.com/senna/

Dependencies
------------

``nlpnet for Korean``은 NLTK_ 와 numpy_ 를 필요로 합니다. 또, NLTK를 인스톨한 후에, 추가적인 data를 다운로드 받아야 합니다. 인스톨 후에 아래의 명령어를 호출하십시오.

    >>> nltk.download()

후에, `Models` 탭으로 가서, `Punkt tokenizer`를 선택합니다. `Punkt tokenizer`는 텍스트를 문장으로 나누는데 사용됩니다.

Cython_ 은 C extension을 생성하고, 빠르게 실행시키기 위해 사용됩니다.
``.c``파일을 사용하는 ``nlpnet for Korean``은 그렇기에 Cython_ 에 종속적이고, ``C 컴파일러``가 필요합니다.
setuptools_ 는 또한 Microsoft의 C 컴파일러를 필요로 하기 때문에, Linux나 OSX환경에서는 문제가 되지 않지만, Windows 사용자라면 곤란할 수 있습니다.
Microsoft의 C 컴파일러를 설치하기 전이라면, MinGW_ 를 사용하는 것이 빠를 수 있습니다.

.. _NLTK: http://www.nltk.org
.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org
.. _MinGW: http://www.mingw.org
.. _setuptools: http://pythonhosted.org/setuptools/

Basic usage
-----------

``nlpnet`` can be used both as a Python library or by its standalone scripts. Both usages are explained below.

Library usage
~~~~~~~~~~~~~

You can use ``nlpnet`` as a library in Python code as follows:

.. code-block:: python

    >>> import nlpnet
    >>> tagger = nlpnet.POSTagger('/path/to/pos-model/', language='pt')
    >>> tagger.tag('O rato roeu a roupa do rei de Roma.')
    [[(u'O', u'ART'), (u'rato', u'N'), (u'roeu', u'V'), (u'a', u'ART'), (u'roupa', u'N'), (u'do', u'PREP+ART'), (u'rei', u'N'), (u'de', u'PREP'), (u'Roma', u'NPROP'), (u'.', 'PU')]]

In the example above, the ``POSTagger`` constructor receives as the first argument the directory where its trained model is located. The second argument is the two letter language code (currently, onle ``pt`` and ``en`` are supported). This only has impact in tokenization.

Calling an annotation tool is pretty straightforward. The provided ones are ``POSTagger``, ``SRLTagger`` and ``DependencyParser``, all of them having a method ``tag`` which receives strings with text to be tagged (in ``DependencyParser``, there is an alias to the method ``parse``, which sounds more adequate). The tagger splits the text into sentences and then tokenizes each one (hence the return of the POSTagger is a list of lists).

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

Here's an example with the DependencyParser:

    >>> parser = nlpnet.DependencyParser('dependency', language='en')
    >>> parsed_text = parser.parse('The book is on the table.')
    >>> parsed_text
    [<nlpnet.taggers.ParsedSentence at 0x10e067f0>]
    >>> sent = parsed_text[0]
    >>> print(sent.to_conll())
    1       The     _       DT      DT      _       2       NMOD
    2       book    _       NN      NN      _       3       SBJ
    3       is      _       VBZ     VBZ     _       0       ROOT
    4       on      _       IN      IN      _       3       LOC-PRD
    5       the     _       DT      DT      _       6       NMOD
    6       table   _       NN      NN      _       4       PMOD
    7       .       _       .       .       _       3       P

The ``to_conll()`` method of ParsedSentence objects prints them in the `CoNLL`_ notation. The tokens, labels and head indices are accessible through member variables:

    >>> sent.tokens
    [u'The', u'book', u'is', u'on', u'the', u'table', u'.']
    >>> sent.heads
    array([ 1,  2, -1,  2,  5,  3,  2])
    >>> sent.labels
    [u'NMOD', u'SBJ', u'ROOT', u'LOC-PRD', u'NMOD', u'PMOD', u'P']
    
The ``heads`` member variable is a numpy array. The i-th position in the array contains the index of the head of the i-th token, except for the root token, which has a head of -1. Notice that these indices are 0-based, while the ones shown in the ``to_conll()`` function are 1-based.

.. _`CoNLL`: http://ilk.uvt.nl/conll/#dataformat

Standalone scripts
~~~~~~~~~~~~~~~~~~

``nlpnet`` also provides scripts for tagging text, training new models and testing them. They are copied to the `scripts` subdirectory of your Python installation, which can be included in the system PATH variable. You can call them from command line and give some text input.

.. code-block:: bash

    $ nlpnet-tag.py pos --data /path/to/nlpnet-data/ --lang pt
    O rato roeu a roupa do rei de Roma.
    O_ART rato_N roeu_V a_ART roupa_N do_PREP+ART rei_N de_PREP Roma_NPROP ._PU

If ``--data`` is not given, the script will search for the trained models in the current directory. ``--lang`` defaults to ``en``. If you have text already tokenized, you may use the ``-t`` option; it assumes tokens are separated by whitespaces.
    
With semantic role labeling:

.. code-block:: bash

    $ nlpnet-tag.py srl /path/to/nlpnet-data/
    O rato roeu a roupa do rei de Roma.
    O rato roeu a roupa do rei de Roma .
    roeu
        A1: a roupa do rei de Roma
        A0: O rato
        V: roeu

The first line was typed by the user, and the second one is the result of tokenization.

And dependency parsing:

.. code-block:: bash

    $ nlpnet-tag.py dependency --data dependency --lang en
    The book is on the table.
    1       The     _       DT      DT      _       2       NMOD
    2       book    _       NN      NN      _       3       SBJ
    3       is      _       VBZ     VBZ     _       0       ROOT
    4       on      _       IN      IN      _       3       LOC-PRD
    5       the     _       DT      DT      _       6       NMOD
    6       table   _       NN      NN      _       4       PMOD
    7       .       _       .       .       _       3       P

To learn more about training and testing new models, and other functionalities, refer to the documentation at http://nilc.icmc.usp.br/nlpnet
