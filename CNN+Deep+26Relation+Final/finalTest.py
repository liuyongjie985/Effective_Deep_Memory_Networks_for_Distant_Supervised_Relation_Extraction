# coding:utf-8

import theano
import theano.tensor as T
import numpy as np

rng = np.random.RandomState(3435)

my_in = 690
my_out = 200
W_bound = np.sqrt(6. / (my_in + my_out))

WForAT_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[my_in, my_out]), dtype=theano.config.floatX)
RForAT_init = np.asarray(rng.uniform(low=-0.01, high=0.01, size=[26, 200]), dtype=theano.config.floatX)

WForAT = theano.shared(value=WForAT_init, name="WForAT")
RForAT = theano.shared(value=RForAT_init, name="RForAT")


# finalInput4 = 26×(690+120)
def finalLayer4(finalInput4):
    # finalInput1 = 26×(690+120)
    def finalLayer1(Si, finalInput1):
        # finalInput2 = 1×(690+120)
        def finalLayer2(finalInput2):
            aabb = T.dot(finalInput2, WForFianl)
            bbcc = T.dot(aabb, finalInput1[Si])
            return bbcc

        # 26×1
        e_fianlre1, _ = theano.scan(finalLayer2, sequences=[finalInput1])

        a_fianlre1 = T.nnet.softmax(e_fianlre1)[0]

        # finalLayer3 = (690+120)
        def finalLayer3(finalInput3, wei):
            return finalInput3 * wei

        # 26×(690+120)
        kak, _ = theano.scan(finalLayer3, sequences=[finalInput1, a_fianlre1])

        # 1×(690+120)
        kaka = T.sum(kak, axis=0)

        # kaka即为目标
        return kaka

    # idx = 0-25
    # 26×(690+120)
    enheng, _ = theano.scan(finalLayer1, sequences=[idx], non_sequences=[finalInput4])
    return enheng


# 50×26×(690+120)
oksecond, _ = theano.scan(finalLayer4, sequences=[])


ep2m = T.imatrix()
cp_out = T.matrix()
idx = T.ivector()

re, up = theano.scan(forEveryRelation, sequences=[idx], non_sequences=[ep2m, cp_out])

myfunc = theano.function([idx, ep2m, cp_out], re)

a = np.asarray([[0, 2], [2, 4]], dtype='int32')
b = np.asarray(rng.uniform(low=-0.01, high=0.01, size=[4, 690]), dtype=theano.config.floatX)
idx_init = np.arange(2, dtype='int32')

ok = myfunc(idx_init, a, b)

print ok


# print WForAT_init
