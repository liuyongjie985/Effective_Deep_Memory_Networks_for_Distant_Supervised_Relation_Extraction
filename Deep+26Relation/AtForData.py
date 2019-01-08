# coding:utf-8
import theano
import theano.tensor as T
import numpy as np

_N_PAD_HEAD = 0
NUMBER_DATA = 3
_IMG_W = 60
rng = np.random.RandomState(3435)
img_h = 88

data_my_in = _IMG_W * 3

W_bound = np.sqrt(1. / (data_my_in))

WForATData_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[data_my_in]),
                             dtype=theano.config.floatX)

WForATData = theano.shared(value=WForATData_init, name="WForAT")

linearW_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[_IMG_W, _IMG_W * 2]),
                          dtype=theano.config.floatX)

linearW = theano.shared(value=linearW_init, name="linearW")

BForATData_init = np.asarray(rng.uniform(low=-1, high=1, size=[img_h - 2]),
                             dtype=theano.config.floatX)
BForATData = theano.shared(value=BForATData_init, name="BForATData")

WForEP_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[_IMG_W * 2, _IMG_W * 2]),
                         dtype=theano.config.floatX)

BForEP_init = np.asarray(rng.uniform(low=-1, high=1, size=[_IMG_W * 2]),
                         dtype=theano.config.floatX)

WForEP = theano.shared(value=WForEP_init, name="WForEP")
BForEP = theano.shared(value=BForEP_init, name="BForEP")


# 从这里出一个60维的向量，放在卷积层后面
# sentence = 1×88×60
def atData(input, left, right, Slen):
    sentence = input[0]

    min = T.switch(T.lt(left, right), left, right)
    max = T.switch(T.lt(left, right), right, left)

    sentenceHead = sentence[:(min + _N_PAD_HEAD)]
    sentenceMiddle = sentence[(min + _N_PAD_HEAD + 1):(max + _N_PAD_HEAD)]
    sentenceTail = sentence[(max + _N_PAD_HEAD + 1):]

    # 去掉了两个entityPair
    # 86×60
    newSentence = T.vertical_stack(sentenceHead, sentenceMiddle, sentenceTail)

    # (Slen-2)×60
    originSentence = newSentence[4:Slen + 2]

    leftEntity = sentence[min + _N_PAD_HEAD]
    rightEntity = sentence[max + _N_PAD_HEAD]

    LRConnect = T.concatenate([leftEntity, rightEntity])

    # def AtLayerData(LRConnect):
    #     def forEveryWord(word):
    #         temp = T.concatenate([word, LRConnect])
    #         # return T.concatenate(temp, rightEntity)
    #         return temp
    #
    #     # 将两个entitypair加在了每个句子的后面
    #     # 86×180
    #     sentenceAfAdd, _ = theano.scan(forEveryWord, sequences=newSentence)
    #
    #     # 86×1
    #     eForWord = T.dot(sentenceAfAdd, WForATData)
    #
    #     eAfterNonL = T.tanh(eForWord + BForATData)
    #     # (Slen - 2)×60
    #     eAfterNonL = eAfterNonL[4:Slen + 2]
    #
    #     # Slen-2×1
    #     aForWord = T.nnet.softmax(eAfterNonL)[0]
    #
    #     def mulWeight(word, weight):
    #         return word * weight
    #
    #     # 句子长度×60
    #     newSRep, _ = theano.scan(mulWeight, sequences=[originSentence, aForWord])
    #
    #     # 1×60
    #     finalSRep = T.sum(newSRep, axis=0)
    #     # 1×120
    #     finSRepAfNon = T.dot(finalSRep, linearW)
    #
    #     finSRepAfNon = finSRepAfNon + T.dot(LRConnect, WForEP) + BForEP
    #
    #     return [finSRepAfNon, newSRep]
    #
    # [finalSRep, myob], _ = theano.scan(AtLayerData, outputs_info=[LRConnect, None], n_steps=NUMBER_DATA)

    # return [finalSRep[-1], myob[-1]]
    return originSentence


input2 = T.dtensor4()

left2 = T.ivector()
right2 = T.ivector()
Slen2 = T.ivector()

input1 = T.dtensor3()

left1 = T.iscalar()
right1 = T.iscalar()
Slen1 = T.iscalar()

# ok = atData(input1, left1, right1, Slen1)
ok, _ = theano.scan(atData, sequences=[input2, left2, right2, Slen2])

myfunc = theano.function([input2, left2, right2, Slen2], ok, on_unused_input='ignore')

input2_init = np.reshape(np.arange(2 * 5280, dtype=theano.config.floatX), (2, 1, 88, 60))
left2_init = np.asarray([1, 2], dtype='int32')
right2_init = np.asarray([60, 59], dtype='int32')
Slen2_init = np.asarray([70, 69], dtype='int32')

input1_init = np.reshape(np.arange(5280, dtype=theano.config.floatX), (1, 88, 60))
left1_init = 1
right1_init = 60
Slen1_init = 70

# ok1, ok2, ok3 = myfunc(input_init, left_init, right_init)
#
# print ok1
# print ok2
# print ok3

# ok1, ok2 = myfunc(input_init, left_init, right_init, Slen_init)
# print ok1.shape
# print ok1
# print ok2.shape
# print ok2

ok1 = myfunc(input2_init, left2_init, right2_init, Slen2_init)
print ok1.shape
print ok1
# print ok2.shape
# print ok2
