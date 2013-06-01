# -*- coding: utf-8 -*-

"""
Auxiliary functions for POS tagging training and 
corpus reading.
"""

from pos.macmorphoreader import MacMorphoReader

def create_reader_pos():
    """
    Creates and returns a TextReader object for the POS tagging task.
    """
    return MacMorphoReader()

if __name__ == '__main__':
    pass







