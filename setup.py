import numpy as np
from setuptools import setup
from setuptools import Extension


def readme():
    with open('README.rst') as f:
        text = f.read()
    return text

setup(
    name='nlpnet',
    description='Neural networks for NLP tasks',
    packages=['nlpnet', 'nlpnet.pos', 'nlpnet.srl', 'nlpnet.parse'],
    ext_modules=[
        Extension(
            "nlpnet.network",
            ["nlpnet/network.c"],
            include_dirs=['.', np.get_include()]
        )
    ],
    scripts=[
        'bin/nlpnet-tag.py',
        'bin/nlpnet-train.py',
        'bin/nlpnet-test.py',
        'bin/nlpnet-load-embeddings.py'
    ],
    install_requires=[
        'numpy>=1.9.0',
        'nltk>=3.2.2',
        'six>=1.10',
        'h5py>=2.8.0rc1'
    ],
    license='MIT',
    version='1.2.4',
    author='Erick Fonseca',
    author_email='erickrfonseca@gmail.com',
    url='http://nilc.icmc.usp.br/nlpnet',
    long_description=readme()
)
