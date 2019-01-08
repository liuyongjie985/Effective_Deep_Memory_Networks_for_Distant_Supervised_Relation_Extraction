# coding:utf-8
import theano
import theano.tensor as T
import numpy as np

ma1 = T.imatrix()
vec = T.ivector()

ok = T.dot(ma1, vec)

func = theano.function([ma1, vec], ok)

ma1_init = np.reshape(np.arange(60, dtype='int32'), (6, 10))
vec_init = np.arange(10, dtype='int32')

print func(ma1_init, vec_init)
