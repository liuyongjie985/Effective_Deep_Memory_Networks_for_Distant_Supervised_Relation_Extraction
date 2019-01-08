# coding:utf-8
import theano
import theano.tensor as T
import numpy as np
import sys

input = T.itensor3()
Slen = T.ivector()


def atData(input):
    originSentence = input
    originSentence = originSentence
    return originSentence


ok, _ = theano.scan(atData, sequences=[input])

myf = theano.function([input], ok)

input_init = np.reshape(np.arange(40, dtype='int32'), (2, 4, 5))
Slen_init = np.asarray([2, 3], dtype='int32')

print myf(input_init)

# ValueError: could not broadcast input array from shape (3,5) into shape (2,5)

# 师兄，报这个错误的原因在于，我两次传入的Slen长度变了，导致originSentence这个变量长度发生了改变，救命啊师兄
