# 从这里出一个60维的向量，放在卷积层后面
# input = 1×88×60

manyi = T.ivector()


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

    leftEntity = sentence[min + _N_PAD_HEAD]
    rightEntity = sentence[max + _N_PAD_HEAD]

    LRConnect = T.concatenate([leftEntity, rightEntity])

    def AtLayerData(LRConnect):
        def forEveryWord(word):
            temp = T.concatenate([word, LRConnect])
            # return T.concatenate(temp, rightEntity)
            return temp

        # 将两个entitypair加在了每个句子的后面
        # 86×180
        sentenceAfAdd, _ = theano.scan(forEveryWord, sequences=newSentence)

        # 86×1
        eForWord = T.dot(sentenceAfAdd, WForATData)

        eAfterNonL = T.tanh(eForWord + BForATData)

        def punish(item, i):
            item = T.switch(T.lt(i, 4), item - 20, item)
            item = T.switch(T.gt(i, Slen + 1), item - 20, item)
            return item

        eAfterPu, _ = theano.scan(punish, sequences=[eAfterNonL, manyi])

        # 86×1
        aForWord = T.nnet.softmax(eAfterPu)[0]

        def mulWeight(word, weight):
            return word * weight

        # 86×60
        newSRep, _ = theano.scan(mulWeight, sequences=[newSentence, aForWord])

        # 1×60
        finalSRep = T.sum(newSRep, axis=0)
        # 1×120
        finSRepAfNon = T.dot(finalSRep, linearW)

        finSRepAfNon = finSRepAfNon + T.dot(LRConnect, WForEP) + BForEP

        return [finSRepAfNon, aForWord]

    [finalSRep, myob], _ = theano.scan(AtLayerData, outputs_info=[LRConnect, None], n_steps=NUMBER_DATA)

    return [finalSRep[-1], myob[-1]]