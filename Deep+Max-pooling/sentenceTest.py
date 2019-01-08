import theano
import theano.tensor as T
import numpy as np

a = T.vector('a')

b = T.vector('b')


def mulweight(x, y):
    return x + y


re, up = theano.scan(mulweight, sequences=[a, b])

myf = theano.function([a, b], re)

ok1= myf(np.arange(10), np.arange(10, 100))

print ok1
