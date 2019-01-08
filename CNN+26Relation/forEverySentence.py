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


def forEveryRelation(idx, ep2m, cp_out):
    def forEveryExample(ep_mr, csmp_input):
        # 取出来的照样是句子数×690的矩阵
        # 这些句子是对应一个实体对的

        input_41ep = csmp_input[ep_mr[0]: ep_mr[1]]

        def forEverySentence(item):
            temp = T.dot(item, WForAT)
            # ???? change this
            re = T.dot(temp, RForAT[idx])
            return re

        slist, noup = theano.scan(forEverySentence, sequences=input_41ep)

        aForRj = T.nnet.softmax(slist)[0]

        def mulWeight(sentence, weight):
            return sentence * weight

        newSRep, noup = theano.scan(mulWeight, sequences=[input_41ep, aForRj])

        finalresult = T.sum(newSRep, axis=0)

        # return finalresult
        # 一次做完吧

        return finalresult

    my_sec_add_out, _ = theano.scan(fn=forEveryExample, sequences=ep2m, non_sequences=cp_out)

    return my_sec_add_out




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
