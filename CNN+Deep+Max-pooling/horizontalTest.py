import theano
import theano.tensor as T
import numpy as np

a = T.imatrix()
b = T.imatrix()

ok = T.horizontal_stack(a, b)

myfunc = theano.function([a, b], ok)

a_init = np.reshape(np.arange(10, dtype='int32'), (2, 5))
b_init = np.reshape(np.arange(10, 20, dtype='int32'), (2, 5))

ok = myfunc(a_init, b_init)

print ok
