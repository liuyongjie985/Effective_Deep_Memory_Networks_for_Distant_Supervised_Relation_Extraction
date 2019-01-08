import theano
import theano.tensor as T
import numpy as np

test = T.matrix('test')

list = []

re, up = theano.scan(lambda a: a + 1, sequences=test)

re1 = re + 1

list.append(re)
list.append(re1)

fi = T.concatenate(list, axis=1)

fi2 = T.max(fi, axis=0)

myf = theano.function([test], [fi, fi2])

ok1, ok2 = myf(np.reshape(np.arange(10), (2, 5)))

print ok1
print ok2
