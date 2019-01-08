import theano
import theano.tensor as T
import numpy as np

abc = T.matrix()
vc = T.ivector()


def add(vc, item):
    vc = vc + 1
    return item + 1


re, up = theano.scan(add, sequences=vc, outputs_info=abc)

myf = theano.function([vc, abc], re)

ok = myf(np.zeros((3), dtype='int32'), np.reshape(np.arange(10), (2, 5)))

print ok
