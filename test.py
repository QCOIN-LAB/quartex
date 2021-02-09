
import torch 
import torch.nn as nn
import torch.nn.functional as F


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

from collections import OrderedDict

class MyParameter(torch.Tensor):
    r"""A kind of Tensor that is to be considered a module parameter.

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

    def __new__(cls, data=None, requires_grad=True):
        if data is None:
            data = torch.Tensor()
        return torch.Tensor._make_subclass(cls, data, requires_grad)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            result = type(self)(self.data.clone(memory_format=torch.preserve_format), self.requires_grad)
            memo[id(self)] = result
            return result

    def __repr__(self):
        return 'Parameter containing:\n' + super(MyParameter, self).__repr__()

    def __reduce_ex__(self, proto):
        # See Note [Don't serialize hooks]
        return (
            torch._utils._rebuild_parameter,
            (self.data, self.requires_grad, OrderedDict())
        )

    def add_contraint(self, constraint):
        self.constraint = constraint

    def apply_constraint(self):
        if getattr(self, 'constraint'):
            self.data = self.constraint(self.data)


w = MyParameter(torch.ones(2, 3))
w.add_contraint(RangeConstraint())
print(w)
optim = torch.optim.SGD([w], 0.001)

x = torch.rand(4, 2)
y = torch.matmul(x, w)
l = torch.sum(y**2).backward()
optim.step()
print(w)
w.apply_constraint()
print(w)