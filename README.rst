===============================================================
``nlpnet for Korean`` --- Korean Natural Language Processing with neural networks
===============================================================

``nlpnet for Korean`` 은 ``neural network`` 를 이용한 한국어 자연어 처리용 파이썬 라이브러리입니다.
현재, 기존의 nlpnet_ 프로젝트를 가져와 POS만 한글에 맞게 수정한 상태입니다.
앞으로 nlpnet_ 의 SRL과 DEPENDENCY Parsing부분도 수정해야 할 것입니다.
기존의 nlpnet_ 프로젝트의 설명에 따르면 거의 대부분의 구조가 언어에 종속적이지 않으나, 몇몇 함수들은 포르투갈어에 맞게 설계되었다고 합니다.
또한 SENNA_ 에 영감을 받아 시작하였다고도 서술하고 있습니다.

이 페이지에는 일부 오역이 포함되어 있을 수 있습니다. 원문은 nlpnet_ 에서 확인해 주세요.

.. _nlpnet: https://github.com/erickrf/nlpnet/
.. _SENNA: http://ronan.collobert.com/senna/

Dependencies
------------

``nlpnet for Korean`` 은 NLTK_ 와 numpy_ 를 필요로 합니다. 또, NLTK를 인스톨한 후에, 추가적인 data를 다운로드 받아야 합니다. 인스톨 후에 아래의 명령어를 호출하십시오.

    >>> nltk.download()

후에, `Models` 탭으로 가서, `Punkt tokenizer` 를 선택합니다. `Punkt tokenizer` 는 텍스트를 문장으로 나누는데 사용됩니다.

Cython_ 은 C extension을 생성하고, 빠르게 실행시키기 위해 사용됩니다.
``.c`` 파일을 사용하는 ``nlpnet for Korean`` 은 그렇기에 Cython_ 에 종속적이고, ``C 컴파일러`` 가 필요합니다.
setuptools_ 는 또한 Microsoft의 C 컴파일러를 필요로 하기 때문에, Linux나 OSX환경에서는 문제가 되지 않지만, Windows 사용자라면 곤란할 수 있습니다.
Microsoft의 C 컴파일러를 설치하기 전이라면, MinGW_ 를 사용하는 것이 빠를 수 있습니다.

.. _NLTK: http://www.nltk.org
.. _numpy: http://www.numpy.org
.. _Cython: http://cython.org
.. _MinGW: http://www.mingw.org
.. _setuptools: http://pythonhosted.org/setuptools/

Basic usage
-----------

``nlpnet for Korean`` 은 파이썬 라이브러리로, 혹은 `Standalone` 으로 사용할 수 있습니다. 아래에서 두가지 사용방법에 대해서 설명하겠습니다.

Library usage
~~~~~~~~~~~~~

``nlpnet for Korean`` 을 파이썬 라이브러리로 사용하기 위한 코드는 다음과 같습니다.

.. code-block:: python

    >>> import nlpnet
    >>> tagger = nlpnet.POSTagger('/path/to/pos-model-directory/')
    >>> tagger.tag('나는 집에 갔다.')
    [[(u'나는', u'NPJ'), (u'집에', u'NPJ'), (u'갔다', u'NPJ'), (u'.', u'S')]]

위 예제에서 ``POSTagger`` 생성자는 첫번째 인자로 모델의 위치를 전달해주면 됩니다. (기본설정 : nlpnet/bin/)
태거를 호출해 사용하는 것이 꽤나 간단합니다. ``POSTagger`` , ``SRLTagger`` , ``DependencyParser`` 를 지원할 예정입니다.
모든 모델은 ``tag`` 메소드를 사용해서 태거를 동작시킬 수 있습니다.(``DependencyParser`` 의 경우 ``parse`` .)
태거는 주어진 텍스트를 문장으로 분리한 후, 분석해 리스트로 결과를 내어줍니다.

``SRLTagger`` 의 결과는 조금 더 복잡합니다.

    >>> tagger = nlpnet.SRLTagger()
    >>> tagger.tag(u'로마는 하루아침에 세워진 것이 아니다.')
    [<nlpnet.taggers.SRLAnnotatedSentence at 0x84020f0>]

튜플의 리스트가 아니라, 두 가지 `attribute` 를 가진 `instance` 가 리턴됩니다. 

    >>> sent = tagger.tag(u'로마는 하루아침에 세워진 것이 아니다.')[0]
    >>> sent.tokens
    [u'로마는', u'하루아침에', u'세워진', u'것이', u'아니다', u'.']
    >>> sent.arg_structures
    [(u'아니다',
      {u'A0': [u'로마는'],
       u'A1': [u'것이'],
       u'V': [u'아니다']})]

SRL의 ``argument_structure`` 는 문장내 모든 용언-아규먼트 구조를 담고 있는 리스트입니다.
이번 예제에서는 '아니다'라는 용언에 대해서만을 보인 것입니다.

용언이 튜플의 가장 처음에 나타났고, 'V'의 value에도 다시한번 나타난 것에 유의하시기 바랍니다.

다음은 ``Dependency Parser`` 예제입니다.

    >>> parser = nlpnet.DependencyParser('dependency')
    >>> parsed_text = parser.parse('플라스틱으로 만든 샤베트기는 수입품이 대부분이다.')
    >>> parsed_text
    [<nlpnet.taggers.ParsedSentence at 0x10e067f0>]
    >>> sent = parsed_text[0]
    >>> print(sent.to_conll())
    1      2      NP_AJT      플라스틱/NNG+으로/JKB
    2      3      VP_MOD      만들/VV+ㄴ/ETM
    3      5      NP_SBJ      샤베트기/NNG+는/JX
    4      5      NP_SBJ      수입품/NNG+이/JKS
    5      5      VNP         대부분/NNG+이/VCP+다/EF+./SF

Parsed object에 ``to_conll()`` 메소드를 호출하면 `CoNLL`_ 형식으로 결과를 보여줍니다.
또한 멤버변수에 직접 접근할 수도 있습니다.

    >>> sent.tokens
    [u'플라스틱으로', u'만든', u'샤베트기는', u'수입품이', u'대부분이다', u'.']
    >>> sent.heads
    array([ 2,  3, 5,  5,  5])
    >>> sent.labels
    [u'NP_AJT', u'VP_MOD', u'NP_SBJ', u'NP_SBJ', u'VNP']
    
``heads`` 는 numpy array입니다.
각각의 값은 i번째 어절을 지배소로 가진다는 의미입니다.

.. _`CoNLL`: http://ilk.uvt.nl/conll/#dataformat

Standalone scripts
~~~~~~~~~~~~~~~~~~

``nlpnet`` also provides scripts for tagging text, training new models and testing them.
``nlpnet for Korean`` 은 스크립트로도 사용할 수 있습니다.

아래와 같이 입력하여 결과를 얻을 수 있습니다.

.. code-block:: bash

    $ nlpnet-tag.py pos --data /path/to/nlpnet-data/
    나는 집에 갔다.
    나는_NPJ 집에_NPJ 갔다_VPE ._S

``--data`` 가 주어지지 않은 경우, 스크립트는 현재 디렉토리에서 학습모델을 검색할 것입니다.
문장이 이미 토큰화(토크나이즈) 완료된 경우라면, ``-t`` 옵션을 사용하세요. 띄어쓰기를 토큰 단위로 인식할 것입니다.

Semantic Role Labeling하기.

.. code-block:: bash

    $ nlpnet-tag.py srl /path/to/nlpnet-data/
    나는 집에 갔다.
    나는 집에 갔다.
    갔다.
        A1: 나는
        V: 갔다.

첫번째 열은 유저에게서 입력된 것이고, 두번째는 토큰화 결과입니다.

구문 분석하기.

.. code-block:: bash

    $ nlpnet-tag.py dependency --data dependency
    나는 집에 갔다.
    1   3   NP_SBJ  나/NP+는/JX
    2   3   NP_AJT  집/NNG+에/JKB
    3   3   VP      가/VV+았/EP+다/EF+./SF

    
