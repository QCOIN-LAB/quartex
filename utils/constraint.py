# -*- coding: utf-8 -*-

import torch 
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class RangeConstraint:
    def __init__(self, low=-0.1, high=0.1):
        self.low = low
        self.high = high

    def __call__(self, x):
        return x.clamp_(self.low, self.high)

class UnitNormConstraint:
    def __init__(self):
        pass

    def __call__(self, x):
        return F.normalize(x, p=2, dim=-1)


class ConstrainedParameter(nn.Parameter):
    r"""A constained verision of original parameter, which is actually 
    a kind of Tensor that is to be considered a module parameter.

    Parameters are :class:`~torch.Tensor` subclasses, that have a
    very special property when used with :class:`Module` s - when they're
    assigned as Module attributes they are automatically added to the list of
    its parameters, and will appear e.g. in :meth:`~Module.parameters` iterator.
    Assigning a Tensor doesn't have such effect. This is because one might
    want to cache some temporary state, like last hidden state of the RNN, in
    the model. If there was no such class as :class:`Parameter`, these
    temporaries would get registered too.

    Arguments:
        data (Tensor): parameter tensor.
        requires_grad (bool, optional): if the parameter requires gradient. See
            :ref:`excluding-subgraphs` for more details. Default: `True`
    """

    def add_contraint(self, constraint):
        self.constraint = constraint

    def apply_constraint(self):
        self.data = self.constraint(self.data)