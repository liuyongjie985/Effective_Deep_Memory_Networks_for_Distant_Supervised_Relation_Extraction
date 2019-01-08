import theano
import theano.tensor as T
import numpy as np

t = T.dmatrix()

ok = T.sum(t, axis=1)

myfunc = theano.function([t], ok)

t_init = np.reshape(np.arange(4, dtype=theano.config.floatX), (2, 2))
print myfunc(t_init)
