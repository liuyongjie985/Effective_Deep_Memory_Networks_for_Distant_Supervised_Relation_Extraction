import theano
import theano.tensor as T
import numpy as np

abc = T.matrix()

re, up = theano.scan(lambda a: a + 1, sequences=abc)

re1 = T.sum(re, axis=0)

myf = theano.function([abc], [re, re1])

ok1, ok2 = myf(np.reshape(np.arange(10), (2, 5)))

print ok1
print ok2
