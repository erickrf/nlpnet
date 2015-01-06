# -*- coding: utf-8 -*-

"""
Auxiliary functions for POS tagging training and 
corpus reading.
"""

from .pos_reader import POSReader

def create_reader_pos():
    """
    Creates and returns a TextReader object for the POS tagging task.
    """
    return POSReader()

if __name__ == '__main__':
    pass







