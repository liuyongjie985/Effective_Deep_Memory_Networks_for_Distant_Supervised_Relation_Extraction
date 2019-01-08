# coding:utf-8

import theano
import theano.tensor as T
import numpy as np
import logging
import lasagne
from NAas0s_jxt_model import compute_cost
from NAas0s_jxt_model import predict_relations

from theano_layers import LeNetConvPoolLayer, MyMLPDropout, TwoConvPoolLayers

rng = np.random.RandomState(3435)

my_in = 690
my_out = 200
W_bound = np.sqrt(6. / (my_in + my_out))

WForAT_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[my_in, my_out]), dtype=theano.config.floatX)
RForAT_init = np.asarray(rng.uniform(low=-0.01, high=0.01, size=[26, 200]), dtype=theano.config.floatX)

WForAT = theano.shared(value=WForAT_init, name="WForAT")
RForAT = theano.shared(value=RForAT_init, name="RForAT")


def Iden(x): return x


_MLP_ACTIVATIONS = [Iden]

_BATCH_SIZE = 40

_DROPOUT_RATES = [0.5]

logging.info('  - Defining MLP layer')

_USE_STACK_CP = False
_USE_PIECEWISE_POOLING_41CP = True
_MLP_SHAPE = [230, 2]
if not _USE_STACK_CP and _USE_PIECEWISE_POOLING_41CP:
    # _MLP_SHAPE = [230, 26]
    _MLP_SHAPE[0] *= 3
    # _MLP_SHAPE = [690, 26]
    logging.info("    - MLP shape changes to {0}, because of piecewise max-pooling".format(_MLP_SHAPE))

# _MLP_SHAPE = [690,26] _MLP_ACTIVATIONS = [Iden] dropout_rates = [0.5]
my_mlp_layer = MyMLPDropout(rng, layer_sizes=_MLP_SHAPE, activations=_MLP_ACTIVATIONS, dropout_rates=_DROPOUT_RATES)


def forEveryRelation(idx, otheridx, ep2m, cp_out):
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

    # input_shape = (batch_size = 50,_MLP_SHAPE[0] = 690)
    normalre, dropoutre = my_mlp_layer.feed(otheridx, my_sec_add_out, input_shape=(_BATCH_SIZE, _MLP_SHAPE[0]))

    return [normalre, dropoutre]


idx = T.ivector()
otheridx = T.ivector()

ep2m = T.imatrix()
cp_out = T.matrix()

[normalre, dropoutre], up = theano.scan(forEveryRelation, sequences=[idx],
                                        non_sequences=[otheridx, ep2m, cp_out])

fiNormalre = T.transpose(normalre)
fiDropoutre = T.transpose(dropoutre)

dropout_p_ygx_batch = fiDropoutre
p_ygx_batch = fiNormalre
ys = T.matrix('ys', dtype='int32')

dropout_cost = compute_cost(dropout_p_ygx_batch, ys)

cost = compute_cost(p_ygx_batch, ys)

grad_updates = lasagne.updates.adadelta(dropout_cost, my_mlp_layer.params)

predictions = predict_relations(p_ygx_batch)

pred_pscores = p_ygx_batch

# myfunc = theano.function([idx, otheridx, ep2m, cp_out], [fiNormalre, fiDropoutre])

myfunc = theano.function([idx, otheridx, ep2m, cp_out, ys], [fiNormalre,fiDropoutre],
                         updates=grad_updates)

ep2m_init = np.asarray([[0, 2], [2, 4], [4, 5]], dtype='int32')
c_out_init = np.asarray(rng.uniform(low=-0.01, high=0.01, size=[5, 690]), dtype=theano.config.floatX)
idx_init = np.arange(26, dtype='int32')
otheridx_init = np.arange(26, dtype='int32')

yl1 = [0] * 26
yl1[2] = 1
yl2 = [0] * 26
yl2[5] = 1
yl3 = [0] * 26
yl3[7] = 1

ys_init = np.asarray([yl1, yl2, yl3], dtype='int32')

# ok1, ok2 = myfunc(idx_init, otheridx_init, a, b)
for i in xrange(10):
    ok1, ok2= myfunc(idx_init, otheridx_init, ep2m_init, c_out_init, ys_init)
    print 'normalre'
    print ok1.shape
    print 'dropoutre'
    print ok2.shape



# print myfunc(idx_init, otheridx_init, a, b, ys_init)


# print WForAT_init
