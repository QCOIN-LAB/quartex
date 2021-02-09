# -*- coding: utf-8 -*-

from .data_reader import DataReader

def setup(opt):
    reader = DataReader(opt)# DataReader(opt)
    return reader
