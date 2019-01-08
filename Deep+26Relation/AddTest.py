import theano
import theano.tensor as T
import numpy as np

a = T.ivector()
b = T.ivector()

c = a + b

d = T.tanh(c)

myf = theano.function([a, b], d)

print myf(np.arange(10, dtype='int32'), np.arange(10, 20, dtype='int32')).shape
