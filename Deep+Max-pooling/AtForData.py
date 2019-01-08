# coding:utf-8
import theano
import theano.tensor as T
import numpy as np

_N_PAD_HEAD = 0
NUMBER_DATA = 3
_IMG_W = 60
rng = np.random.RandomState(3435)

data_my_in = _IMG_W * 3

W_bound = np.sqrt(1. / (data_my_in))

WForATData_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[NUMBER_DATA, data_my_in]),
                             dtype=theano.config.floatX)

WForATData = theano.shared(value=WForATData_init, name="WForAT")


# 从这里出一个60维的向量，放在卷积层后面
# sentence = 1×88×60
def atData(input, left, right, idx_data):
    sentence = input[0]

    min = T.switch(T.lt(left, right), left, right)
    max = T.switch(T.lt(left, right), right, left)

    sentenceHead = sentence[:(min + _N_PAD_HEAD)]
    sentenceMiddle = sentence[(min + _N_PAD_HEAD + 1):(max + _N_PAD_HEAD)]
    sentenceTail = sentence[(max + _N_PAD_HEAD + 1):]

    # 去掉了两个entityPair
    # 86×60
    newSentence = T.vertical_stack(sentenceHead, sentenceMiddle, sentenceTail)

    leftEntity = sentence[min + _N_PAD_HEAD]
    rightEntity = sentence[max + _N_PAD_HEAD]

    def AtLayerData(idx, newSentenceCon):
        def forEveryWord(word):
            temp = T.concatenate([word, leftEntity, rightEntity])
            # return T.concatenate(temp, rightEntity)
            return temp

        # 将两个entitypair加在了每个句子的后面
        # 86×180
        sentenceAfAdd, _ = theano.scan(forEveryWord, sequences=newSentenceCon)

        eForWord = T.dot(sentenceAfAdd, WForATData[idx])

        aForWord = T.nnet.softmax(eForWord)[0]

        def mulWeight(word, weight):
            return word * weight

        newSRep, _ = theano.scan(mulWeight, sequences=[newSentence, aForWord])

        return newSRep

    finalSRep, _ = theano.scan(AtLayerData, sequences=idx_data, outputs_info=newSentence)

    return T.sum(finalSRep[-1], axis=0)


input = T.dtensor3()

left = T.iscalar()
right = T.iscalar()
idx_data = T.ivector()

ok = atData(input, left, right, idx_data)

myfunc = theano.function([input, left, right, idx_data], ok)

input_init = np.reshape(np.arange(5280, dtype=theano.config.floatX), (1, 88, 60))
left_init = 0
right_init = 87
idx_data_init = np.arange(NUMBER_DATA, dtype='int32')
# ok1, ok2, ok3 = myfunc(input_init, left_init, right_init)
#
# print ok1
# print ok2
# print ok3

ok = myfunc(input_init, left_init, right_init, idx_data_init)
print ok.shape
print ok
