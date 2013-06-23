This system performs Natural Language Processing (NLP) tasks using neural networks and space vector models (VSM) as input.
It was inspired by the SENNA system by Collobert and Weston (2008), but the focuses on Portuguese (it can be used on other languages too, though). Currently, it performs Semantic Role Labeling (SRL) and Part-of-Speech (POS) tagging (the POS models will still be improved).

RUNNING THE TAGGER
==================

If you just want to use the pre-trained the models, follow these steps (should work on any OS with Python):

1. Unzip the contents of the data directory. 
2. Make sure you have python 2.7 with NLTK and numpy installed (you can check it by typing `import nltk` and `import numpy` in the interpreter). If you lack either of these, get them at http://nltk.org/ and http://www.numpy.org/.
3. Make sure you have the Punkt sentence tokenizer from NLTK (used to split the text into sentences). Here's how to do it in the interpreter:


        >>> from nltk import download  
        >>> download()

And then choose the Punkt Tokenizer Models under the Models tab.

Now run run.py from the command line:

$ python run.py --task [TASK]

Where [TASK] means either pos or srl. The system works interactively, but you may provide an input and/or output file using the standard syntax:

$ python run.py --task [TASK] < input.txt > output.txt

Note that the system expects and returns text in UTF-8, so be sure to use this encoding. You may also run run.py with the -h flag to see more optional arguments. If you have any problem with the nlpnet module, see the section below -- perhaps you'll have to recompile it.

COMPILING THE NLPNET MODULE
===========================

The nlpnet module, which includes the actual neural networks, is written in Cython, a superset of Python for easily building C extensions. It is contained in the binary file nlpnet.so (in Linux) or nlpnet.pyd (in Windows). A pre-compiled version of the modules (for 64 bits) is already included to make it easier for users. If you need or want it (you will need if you are using a 32 bits Python interpreter), you can compile from source. You will need Cython (get it from http://www.cython.org/) and a C compiler. 

N.B.: By default, it will need Microsoft Visual Studio in Windows, but generally running gcc under mingw is easier. See http://wiki.cython.org/InstallingOnWindows. 

After installing Cython, you can compile the nlpnet module with:

$ python setup.py build_ext --inplace

This will generate the compiled file in the current directory, and won't install anything in your system (thus, you won't need administrator privileges).

TRAINING
========

Training is a bit more complex than running trained models, as there are many more parameters to adjust. I recommend running `python train.py -h` to see the options. If you want to use the models for other languages, you'll have to change some things in the code, like the expected input format and a few preprocessing steps specific for Portuguese (like contractions).
