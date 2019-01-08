import theano
import theano.tensor as T
import numpy as np

a = T.dvector()

b = T.nnet.softmax(a)[0]

myf = theano.function([a], b)

ok1 = myf(np.arange(10))

print ok1
