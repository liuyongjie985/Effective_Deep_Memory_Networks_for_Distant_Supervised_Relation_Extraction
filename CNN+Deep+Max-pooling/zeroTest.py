import theano
import theano.tensor as T
import numpy as np

test = T.ivector('test')

re, up = theano.scan(lambda a: T.switch(T.eq(a, 0), a + 1, -a), sequences=test)

myf = theano.function([test], re)

ok1 = myf(np.arange(10, dtype='int32'))

print ok1
