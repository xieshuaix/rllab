from rllab.core.serializable import Serializable
from .box_native import BoxNative
import numpy as np
from rllab.misc import ext
import theano


class Box(BoxNative):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    """
    def __init__(self, low, high, shape=None):
        super(Box, self).__init__(low=low, high=high, shape=shape)

    def new_tensor_variable(self, name, extra_dims):
        return ext.new_tensor(
            name=name,
            ndim=extra_dims+1,
            dtype=theano.config.floatX
        )

