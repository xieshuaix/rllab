import theano.tensor.nnet
import theano.tensor as TT
import theano.tensor.extra_ops


def softmax_sym(x):
    return theano.tensor.nnet.softmax(x)


def to_onehot_sym(ind, dim):
    assert ind.ndim == 1
    return theano.tensor.extra_ops.to_one_hot(ind, dim)


def normalize_updates(old_mean, old_std, new_mean, new_std, old_W, old_b):
    """
    Compute the updates for normalizing the last (linear) layer of a neural
    network
    """
    # Make necessary transformation so that
    # (W_old * h + b_old) * std_old + mean_old == \
    #   (W_new * h + b_new) * std_new + mean_new
    new_W = old_W * old_std[0] / (new_std[0] + 1e-6)
    new_b = (old_b * old_std[0] + old_mean[0] - new_mean[0]) / (new_std[0] + 1e-6)
    return OrderedDict([
        (old_W, TT.cast(new_W, old_W.dtype)),
        (old_b, TT.cast(new_b, old_b.dtype)),
        (old_mean, new_mean),
        (old_std, new_std),
    ])

