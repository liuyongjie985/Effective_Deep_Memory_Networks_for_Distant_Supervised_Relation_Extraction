import numpy as np
import theano.tensor as T
import theano

a = T.dmatrix()
b = T.nnet.softmax(a)

myf = theano.function([a], b)

t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

t = np.asarray(t)

print myf(t)

c = T.dvector()
d = T.nnet.softmax(c)

myf = theano.function([c], d)

print myf(np.arange(10))

a = [1, 2, 4, 5, 6, 7, 8, 9, 0, 1]

print a[:1]
print a[1:3]
