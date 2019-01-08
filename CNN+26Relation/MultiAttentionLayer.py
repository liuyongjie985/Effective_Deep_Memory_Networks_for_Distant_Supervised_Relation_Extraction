# coding:utf-8
import theano
import numpy as np
import theano.tensor as T

rng = np.random.RandomState(3435)

my_in = 690
my_out = 200
W_bound = np.sqrt(6. / (my_in + my_out))

WForAT_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[my_in, my_out]), dtype=theano.config.floatX)
RForAT_init = np.asarray(rng.uniform(low=-0.01, high=0.01, size=[26, 200]), dtype=theano.config.floatX)

WForAT = theano.shared(value=WForAT_init, name="WForAT")
RForAT = theano.shared(value=RForAT_init, name="RForAT")

input_41ep = T.matrix()
number = T.iscalar()


# 取出来的 句子数×690矩阵
def attentionLayer(input_41ep_out):
    def forEverySentence(item):
        temp = T.dot(item, WForAT)
        # ???? change this
        re = T.dot(temp, RForAT[0])
        return re

    # slist就是ei
    slist, noup = theano.scan(forEverySentence, sequences=input_41ep_out)

    aForRj = T.nnet.softmax(slist)[0]

    def mulWeight(sentence, weight):
        return sentence * weight

    # 句子数×690
    newSRep, noup = theano.scan(mulWeight, sequences=[input_41ep_out, aForRj])

    return newSRep

newSRepAf, _ = theano.scan(attentionLayer, outputs_info=input_41ep, n_steps=number)

myfunc = theano.function([input_41ep, number], newSRepAf)

input_41ep_init  = np.asarray(rng.uniform(low=-0.01, high=0.01, size=[2, 690]), dtype=theano.config.floatX)
number_init = 4

ok = myfunc(input_41ep_init,number_init)

print ok.shape
print ok
