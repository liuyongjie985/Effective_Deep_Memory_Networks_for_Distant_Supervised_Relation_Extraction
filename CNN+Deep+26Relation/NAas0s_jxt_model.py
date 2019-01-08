# -*- coding: utf-8 -*-

import cPickle
import numpy as np

np.set_printoptions(threshold='nan', linewidth='nan')
from collections import OrderedDict
import os
import shutil
import theano
import theano.tensor as T
from theano.ifelse import ifelse
import lasagne
import logging
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from theano_layers import MyLeNetConvPoolLayer, MyMLPDropout, TwoConvPoolLayers
import sys
import datetime
import ConfigParser
import argparse

# ------ 配置文件 ------
conf_filename = "ModelControl.ini"
conf = ConfigParser.ConfigParser()
conf.read(conf_filename)
# assert conf.getboolean('process_dataset', 'merge_ep') == True

# ------ Theano ------
theano.config.floatX = conf.get('theano_setup', 'floatX')
theano.config.exception_verbosity = 'high'

OP_ATT_SEP = '\t\t\t'

# ------ 子文件夹 -------      (保存数据所使用的各个文件夹的位置. 注意这个位置会被改变一次(加上当前时间作为文件夹的名称).)
_DATASETS_READY_DIR = conf.get('file_dir_path', 'datasets_ready_dir')
_RESULT_DATA_DIR = conf.get('file_dir_path', 'result_data_dir')
_PRCURVE_DATA_DIR = conf.get('file_dir_path', 'PRCurve_data_dir')
_OBSERVATION_DIR = conf.get('file_dir_path', 'observation_dir')


# ------- 非线性层 -------
def Iden(x): return x


# -----  设置超参数等 -----
_USE_STACK_CP = conf.getboolean('mode', 'use_stacked_cp')
_USE_PIECEWISE_POOLING_41CP = True
_N_RELATIONS = conf.getint('process_dataset', 'n_relations') - 1
_CP1_FILTER_HS = eval(conf.get('hpyer_parameters', 'cp1_filter_hs'))
_CP2_FILTER_H = conf.getint('hpyer_parameters', 'cp2_filter_h')
_CP1_N_FILTERS = conf.getint('hpyer_parameters', 'cp1_n_filters')
_CP2_N_FILTERS = conf.getint('hpyer_parameters', 'cp2_n_filters')
_CP1_NON_LINEAR = conf.get('hpyer_parameters', 'cp1_non_linear')
_CP2_NON_LINEAR = conf.get('hpyer_parameters', 'cp2_non_linear')

_CP1_POOL_SIZE_4SCP = eval(conf.get('hpyer_parameters', 'cp1_poo1_size_4_stacked_cp'))

_MLP_ACTIVATIONS = [Iden]
# use_stacked_cp = False
# n_mlp_input = 230
n_mlp_input = _CP2_N_FILTERS if _USE_STACK_CP else _CP1_N_FILTERS
# mlp_hidden_layers = []
mlp_hidden_layers = eval(conf.get('hpyer_parameters', 'n_mlp_hidden_layer'))

# _MLP_SHAPE = [230, 26]
# _MLP_SHAPE = [n_mlp_input] + mlp_hidden_layers + [_N_RELATIONS]

_MLP_SHAPE = [230, 2]

_DROPOUT_RATES = eval(conf.get('hpyer_parameters', 'dropout_rates'))
_N_EPOCHS = conf.getint('hpyer_parameters', 'n_epochs')
_BATCH_SIZE = conf.getint('hpyer_parameters', 'batch_size')
_SHUFFLE_BATCH = conf.getboolean('hpyer_parameters', 'shuffle_batch')
_LR = conf.getfloat('hpyer_parameters', 'learning_rate')
_LR_DECAY = conf.getfloat('hpyer_parameters', 'lr_decay')
_SQR_NORM_LIM = conf.getint('hpyer_parameters', 'sqr_norm_lim')
_LEN_WORDV = conf.getint('hpyer_parameters', 'wordv_length')
_LEN_PFV = conf.getint('hpyer_parameters', 'pfv_length')
_IMG_W = _LEN_WORDV + 2 * _LEN_PFV
_CP1_FILTER_W = _IMG_W
_N_PAD_HEAD = conf.getint('settings', 'max_filter_h') - 1

NUMBER = conf.getint('hpyer_parameters', 'numForAt')
NUMBER_DATA = conf.getint('hpyer_parameters', 'numForAtData')

rng = np.random.RandomState(3435)


def compute_cost(p_ygx_s, ys):
    """
    计算损失函数
    :param ys: 真实值(符号)
    :return: cost值(符号)
    """

    # cost_type = sigmoid
    cost_type = conf.get('mode', 'cost_type')

    if cost_type == 'min_squared':
        single_cost = T.sum(T.square(p_ygx_s - ys), axis=1)
        cost = T.sum(single_cost)
        return T.cast(cost, dtype=theano.config.floatX)

    if cost_type == 'sigmoid':
        single_cost = - (T.batched_dot(ys, T.log(p_ygx_s)) + T.batched_dot(1 - ys, T.log(1 - p_ygx_s)))
        cost = T.sum(single_cost)
        return T.cast(cost, dtype=theano.config.floatX)


def predict_relations(p_y_given_x):
    pred_relations, updates = theano.scan(lambda p_ep: p_ep > 0.5, sequences=[p_y_given_x])
    return pred_relations


# 源代码
# def model_mimlcnn(datasets, Wordv, PF1v, PF2v, img_h):

def model_mimlcnn(datasets, Wordv, PF1v, PF2v, img_h, RForAT, WForAT, WForATData, linearW):
    """
    模型建模.
    :param datasets: 放进来的数据集.
    :param Wordv: "Word/Token - Embedding" 矩阵.
    :param PF1v: "PositionFeature1 - Embedding" 矩阵
    :param PF2v: "PositionFeature2 - Embedding" 矩阵
    :param img_h: Padding之后的句子长度.
    :param RForAT: 关系 - embedding矩阵
    :param WForAT: attention算每个句子权值时需要用到的矩阵
    :return: 建模之后的所有模型
    """

    # 1. 确定超参
    logging.info('-------------- Model Settings --------------------')
    # if word embedding is initialized with 'word2vec', then 'length' is set by the dimension of the word vector automatically;
    # if initialized with 'rand', then the specified value of 'length' is used.
    # is_static = False
    w2v_static = conf.getboolean('word_vector', 'is_static')

    # pfv_length = 5 wordv_length = 50 img_W = 50 + 5 * 2 = 60
    image_shape = (None, 1, img_h, _IMG_W)

    cp1_filter_shapes = []
    cp1_pool_sizes = []

    cp2_filter_shape = None
    cp2_pool_size = None

    assert len(_CP1_FILTER_HS) == 1

    # use_stacked_cp = False
    if not _USE_STACK_CP:
        # _CP1_FILTER_HS = [3]
        for filter_h in _CP1_FILTER_HS:
            # cp1_n_filters = 230
            # _IMG_W = _LEN_WORDV + 2 * _LEN_PFV _CP1_FILTER_W = _IMG_W
            # 230,1,3,60
            # 每次抽取3个句子的特征
            cp1_filter_shapes.append((_CP1_N_FILTERS, 1, filter_h, _CP1_FILTER_W))
            # filter完后的行数，86,1
            cp1_pool_sizes.append((img_h - filter_h + 1, _IMG_W - _CP1_FILTER_W + 1))
    else:

        cp1_filter_shapes.append((_CP1_N_FILTERS, 1, _CP1_FILTER_HS[0], _CP1_FILTER_W))

        cp1_pool_sizes.append(_CP1_POOL_SIZE_4SCP)

        cp2_filter_shape = [_CP2_N_FILTERS, _CP1_N_FILTERS, _CP2_FILTER_H, 1]
        cp1_fm_img_h = image_shape[2] - _CP1_FILTER_HS[0] + 1
        cp2_img_h = int(np.ceil(cp1_fm_img_h / float(cp1_pool_sizes[0][0])))
        cp2_pool_size = [cp2_img_h - _CP2_FILTER_H + 1, 1]

    logging.info('|     - image_shape: {0}'.format(image_shape))
    logging.info('|     - cp1_filter_shapes: {0}'.format(cp1_filter_shapes))
    logging.info('|     - cp1_non_linear: {0}'.format(_CP1_NON_LINEAR))
    logging.info('|     - cp1_pool_sizes: {0}'.format(cp1_pool_sizes))

    if _USE_STACK_CP:
        logging.info('|     - cp2_filter_shape: {0}'.format(cp2_filter_shape))
        logging.info('|     - cp2_non_linear: {0}'.format(_CP2_NON_LINEAR))
        logging.info('|     - cp2_pool_sizes: {0}'.format(cp2_pool_size))

    logging.info('|     - initial mlp_shape: {0}'.format(_MLP_SHAPE))
    logging.info('|     - dropout_rates: {0}'.format(_DROPOUT_RATES))
    logging.info('|     - batch_size: {0}'.format(_BATCH_SIZE))
    logging.info('|     - word_embedding_length: {0}'.format(_LEN_WORDV))
    logging.info('|     - word_embedding_initialization: {0}'.format(conf.get('word_vector', 'initialization')))
    logging.info('|     - word_embedding_static: {0}'.format(w2v_static))
    logging.info('|     - shuffle_batch: {0}'.format(_SHUFFLE_BATCH))
    logging.info('|     - lr_decay: {0}'.format(_LR_DECAY))
    logging.info('|     - sqr_norm_lim: {0}'.format(_SQR_NORM_LIM))
    logging.info('|     - learning_rate: {0}'.format(_LR))
    logging.info('|     - cost_type: {0}'.format(conf.get('mode', 'cost_type')))
    logging.info('|         - pr_margin: {0}'.format(conf.getfloat('mode', 'pr_margin')))
    logging.info('|         - score_margin: {0}'.format(conf.getfloat('mode', 'score_margin')))
    logging.info('|     - prediction_type: larger than 0.5 for each label')
    logging.info('--------------------------------------------------')

    # 2. 计算模型的输入
    logging.info('  - Defining model variables for one mini-batch')

    bch_idx = T.scalar('batch_idx', dtype='int32')
    # bch_idx.tag.test_value = 1


    xs = T.matrix('xs', dtype='int32')

    # 3个句子
    # 不要被test_value欺骗，xs是句子数×88维的
    xs.tag.test_value = np.asarray(
        [[0, 0, 0, 0, 3, 2, 3, 7, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 2, 4, 1, 8, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 3, 1, 6, 5, 3, 0, 0, 0, 0, 0, 0]], dtype='int32')

    pfinfos = T.matrix('pfinfos', dtype='int32')
    pfinfos.tag.test_value = np.array([[4, 2, 1], [5, 1, 3], [5, 0, 1]])

    # p0_in_PFv = 52
    p0_in_PFv = conf.getint("settings", "p0_in_PFv")

    # padding位置信息
    def cal_padded_sentpf(pfinfo_m):

        slen = pfinfo_m[0]
        e1i = pfinfo_m[1]
        e2i = pfinfo_m[2]

        pf1 = T.arange(p0_in_PFv - e1i, p0_in_PFv + (slen - e1i), dtype='int32')

        pf2 = T.arange(p0_in_PFv - e2i, p0_in_PFv + (slen - e2i), dtype='int32')

        # 调整到最小1，最大101,长度与句子长度相同
        clipped_pf1 = T.clip(pf1, 1, 101)
        clipped_pf2 = T.clip(pf2, 1, 101)

        # _N_PAD_HEAD = 4
        pad_head = T.zeros(shape=(_N_PAD_HEAD,), dtype='int32')
        pad_tail = T.zeros(shape=(img_h - _N_PAD_HEAD - slen,), dtype='int32')

        # 把两端列表拼成一段列表
        pf1_padded = T.concatenate([pad_head, clipped_pf1, pad_tail])  # 三部分相加=pad后长度.
        pf2_padded = T.concatenate([pad_head, clipped_pf2, pad_tail])

        return pf1_padded, pf2_padded

    # 句子长度，实体1实体2位置 三种信息的padding，padding后向量长度为88，头和尾里面数字是0，实体1和实体2相对位置部分的数字最小是1最大是101
    # 返回的是[实体1的位置padding]，[实体2的位置padding]，都是88维，前4维都为空，中间长度就是句子长度，剩下的也是0
    (pf1s, pf2s), _ = theano.scan(fn=cal_padded_sentpf, sequences=[pfinfos])

    e1is = pfinfos[:, 1]
    e2is = pfinfos[:, 2]

    # 每次传进来的连续x_m片段都是index从0开始的, 而ep2m依旧是按照数据集中所有的x_m来定位的. 所以在这里让ep2m中的所有元素减一个初始值, 让xs的index和ep2m的每个起始位置对应上.
    ep2m_raw = T.matrix('ep2m_raw', dtype='int32')
    ep2m_raw.tag.test_value = np.asarray([[25, 27], [27, 28]], dtype='int32')

    ep2m = ep2m_raw - ep2m_raw[0][0]
    ep2m.tag.test_value = np.asarray([[0, 2], [2, 3]], dtype='int32')

    ys = T.matrix('ys', dtype='int32')
    # _N_RELATIONS = 26
    yl1 = [0] * _N_RELATIONS
    yl1[2] = 1
    yl2 = [0] * _N_RELATIONS
    yl2[5] = 1
    ys.tag.test_value = np.asarray([yl1, yl2], dtype='int32')

    # 3. 定义模型结构, 定义输出, 定义损失
    assert pf1s.dtype == 'int32' and pf2s.dtype == 'int32' and xs.dtype == 'int32'

    _use_my_input = True

    if _use_my_input:
        # 1. 我的拼接方法
        # 看到这终于明白了，Wordv是用word2Vec初始好的词向量
        # 以维度为1连接传入数据，一个句子本来是向量，现在转成了矩阵
        # Wordv[xs.flatten()] = [50维的词向量]
        # pf1s =
        fltn_vec_stk = T.horizontal_stack(Wordv[xs.flatten()], PF1v[pf1s.flatten()], PF2v[pf2s.flatten()])
        # 句子数×1×88×60
        cp_layer_input = fltn_vec_stk.reshape((xs.shape[0], 1, xs.shape[1], _IMG_W))

    else:
        # 2. Zeng的拼接方法
        input_words = Wordv[xs.flatten()].reshape((xs.shape[0], 1, xs.shape[1], _LEN_WORDV))
        input_pf1s = PF1v[pf1s.flatten()].reshape((pf1s.shape[0], 1, pf1s.shape[1], _LEN_PFV))
        input_pf2s = PF2v[pf2s.flatten()].reshape((pf2s.shape[0], 1, pf2s.shape[1], _LEN_PFV))
        cp_layer_input = T.concatenate([input_words, input_pf1s, input_pf2s], axis=3)

    logging.info('  - Defining and assembling CP layer')
    cp_params = []

    # 句子数×1×88×60
    # cp_layer_input =

    # 从这里出一个60维的向量，放在卷积层后面
    # input = 1×88×60
    def atData(input, left, right):
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

        def AtLayerData(LRConnect, newSentenceCon):
            def forEveryWord(word):
                temp = T.concatenate([word, LRConnect])
                # return T.concatenate(temp, rightEntity)
                return temp

            # 将两个entitypair加在了每个句子的后面
            # 86×180
            sentenceAfAdd, _ = theano.scan(forEveryWord, sequences=newSentenceCon)

            eForWord = T.dot(sentenceAfAdd, WForATData)

            aForWord = T.nnet.softmax(eForWord)[0]

            def mulWeight(word, weight):
                return word * weight

            # 86×60
            newSRep, _ = theano.scan(mulWeight, sequences=[newSentence, aForWord])

            # 1×60
            finalSRep = T.sum(newSRep, axis=0)

            return T.dot(finalSRep, linearW)

        finalSRep, _ = theano.scan(AtLayerData, outputs_info=LRConnect, non_sequences=newSentence, n_steps=NUMBER_DATA)

        return finalSRep[-1]

    myobser1, _ = theano.scan(atData, sequences=[cp_layer_input, e1is, e2is])

    # False
    if _USE_STACK_CP:
        logging.info('    - [!] Use stacked CP layer, NO Piecewise pooling.')
        two_cplayers = TwoConvPoolLayers(
            rng,
            cp1_filter_shape=cp1_filter_shapes[0],
            cp1_pool_size=cp1_pool_sizes[0],
            cp2_filter_shape=cp2_filter_shape,
            cp2_pool_size=cp2_pool_size,
            image_shape=image_shape,
            cp1_nonlinear=_CP1_NON_LINEAR,
            cp2_nonlinear=_CP2_NON_LINEAR,
        )
        two_cplayers.feed(cp_layer_input)
        raw_op = two_cplayers.output
        cp_out = T.addbroadcast(raw_op, 2, 3).dimshuffle(0, 1)
        cp_params = two_cplayers.params
    else:
        logging.info('    - Use a single CP layer')
        cp_layers = []

        # _CP1_FILTER_HS = [3]

        assert len(_CP1_FILTER_HS) == 1

        for i in range(len(_CP1_FILTER_HS)):
            # 有3个filter

            # 230,1,3,60
            # 每次抽取3个句子的特征
            # cp1_filter_shapes.append((_CP1_N_FILTERS, 1, filter_h, _CP1_FILTER_W))

            # filter完后的行数，86,1
            # cp1_pool_sizes.append((img_h - filter_h + 1, _IMG_W - _CP1_FILTER_W + 1))

            # pfv_length = 5 wordv_length = 50 img_W = 50 + 5 * 2 = 60
            # image_shape = (None, 1, img_h, _IMG_W)

            # cp1_non_linear = tanh

            # 初始好卷积层filter的W与B----源码
            # cp_layer = LeNetConvPoolLayer(rng, filter_shape=cp1_filter_shapes[i], poolsize=cp1_pool_sizes[i],
            #                               image_shape=image_shape,
            #                               non_linear=_CP1_NON_LINEAR)
            # cp_layers.append(cp_layer)


            cp_layer = MyLeNetConvPoolLayer(rng, filter_shape=cp1_filter_shapes[i], poolsize=cp1_pool_sizes[i],
                                            image_shape=image_shape,
                                            non_linear=_CP1_NON_LINEAR)
            cp_layers.append(cp_layer)

        chnl_outputs = []

        # _CP1_FILTER_HS = [3]
        myobser = None
        for layer_idx in range(len(_CP1_FILTER_HS)):
            # _USE_PIECEWISE_POOLING_41CP = True
            if _USE_PIECEWISE_POOLING_41CP:  # 1. Piecewise版本
                logging.info('     - Use piecewise max-pooling')

                # 句子数×1×88×60
                # cp_layer_input = fltn_vec_stk.reshape((xs.shape[0], 1, xs.shape[1], _IMG_W))
                # 句子数×1,每个句子实体1下标
                # e1is = pfinfos[:, 1]
                # self.input = cp_layer_input
                # self.output = pooling_2d 对于每一个句子返回一个1×690的向量
                myobser = cp_layers[layer_idx].piecewisePooling_feed([cp_layer_input, e1is, e2is])

                cp_output = cp_layers[layer_idx].output


            else:  # 2. 非Piecewise版本
                logging.info('     - Use normal max-pooling')
                cp_layers[layer_idx].feed(cp_layer_input)
                cp_output_raw = cp_layers[layer_idx].output
                cp_output = T.addbroadcast(cp_output_raw, 2, 3).dimshuffle(0, 1)

            chnl_outputs.append(cp_output)
        # 本是应该把不同层数filter所提取的句子表示叠加起来，但是这里只有一层所以没有效果
        # cp_out 句子数×690
        cp_out = T.concatenate(chnl_outputs, axis=1)

        # cp层就一层,两个参数就是W和b
        for cp_layer in cp_layers:
            cp_params += cp_layer.params

    # ****************
    # *****源代码******
    # *****************
    #
    #
    # def ep_max_pooling(ep_mr, csmp_input):
    #     # 取出来的照样是句子数×690的矩阵
    #     input_41ep = csmp_input[ep_mr[0]: ep_mr[1]]
    #     # Cross-sentence Max-pooling
    #     max_pooling_out = T.max(input_41ep, axis=0)
    #     # 返回的就是 Entity-pair Representation
    #     return max_pooling_out
    #
    # logging.info('  - Aassembling second Max-Pooling layer')
    #
    # # Entity-pair Representation的列表
    # sec_maxPooling_out, _ = theano.scan(fn=ep_max_pooling, sequences=ep2m, non_sequences=cp_out)


    # **************************
    # *****加入attention机制******
    # **************************

    # 针对1个关系

    # def forEveryExample(ep_mr, csmp_input):
    #     # 取出来的照样是句子数×690的矩阵
    #     # 这些句子是对应一个实体对的
    #
    #     input_41ep = csmp_input[ep_mr[0]: ep_mr[1]]
    #
    #     def forEverySentence(item):
    #         temp = T.dot(item, WForAT)
    #         # ???? change this
    #         re = T.dot(temp, RForAT[0])
    #         return re
    #
    #     slist, noup = theano.scan(forEverySentence, sequences=input_41ep)
    #
    #     aForRj = T.nnet.softmax(slist)[0]
    #
    #     def mulWeight(sentence, weight):
    #         return sentence * weight
    #
    #     newSRep, noup = theano.scan(mulWeight, sequences=[input_41ep, aForRj])
    #
    #     finalresult = T.sum(newSRep, axis=0)
    #
    #     # return finalresult
    #     return finalresult
    #
    # # # Entity-pair Representation的列表
    # my_sec_add_out, _ = theano.scan(fn=forEveryExample, sequences=ep2m, non_sequences=cp_out)
    #
    # logging.info('  - Defining MLP layer')
    # # _USE_STACK_CP = False
    # # _USE_PIECEWISE_POOLING_41CP = True
    # if not _USE_STACK_CP and _USE_PIECEWISE_POOLING_41CP:
    #     # _MLP_SHAPE = [230, 26]
    #     _MLP_SHAPE[0] *= 3
    #     # _MLP_SHAPE = [690, 26]
    #     logging.info("    - MLP shape changes to {0}, because of piecewise max-pooling".format(_MLP_SHAPE))
    #
    # # _MLP_SHAPE = [690,26] _MLP_ACTIVATIONS = [Iden] dropout_rates = [0.5]
    # mlp_layer = MLPDropout(rng, layer_sizes=_MLP_SHAPE, activations=_MLP_ACTIVATIONS, dropout_rates=_DROPOUT_RATES)
    # # input_shape = (batch_size = 50,_MLP_SHAPE[0] = 690)
    # mlp_layer.feed(my_sec_add_out, input_shape=(_BATCH_SIZE, _MLP_SHAPE[0]))

    # cp_out = 句子数×690 myobser1 = 句子数×120
    # new_cp_out = 句子数×810
    new_cp_out = T.horizontal_stack(cp_out, myobser1)

    # 针对26个关系

    logging.info('  - Defining MLP layer')

    assert _USE_STACK_CP == False
    assert _USE_PIECEWISE_POOLING_41CP == True

    if not _USE_STACK_CP and _USE_PIECEWISE_POOLING_41CP:
        # _MLP_SHAPE = [230, 26]
        _MLP_SHAPE[0] *= 3
        # _MLP_SHAPE = [690, 26]
        logging.info("    - MLP shape changes to {0}, because of piecewise max-pooling".format(_MLP_SHAPE))

    # _MLP_SHAPE = [690,26] _MLP_ACTIVATIONS = [Iden] dropout_rates = [0.5]


    my_mlp_layer = MyMLPDropout(rng, layer_sizes=[[_MLP_SHAPE[0] + _IMG_W * 2, 2]], activations=_MLP_ACTIVATIONS,
                                dropout_rates=_DROPOUT_RATES)

    def forEveryRelation(idx, ep2m, cp_out):
        def forEveryExample(ep_mr, csmp_input):
            # 取出来的照样是句子数×(690+120)的矩阵
            # 这些句子是对应一个实体对的

            input_41ep = csmp_input[ep_mr[0]: ep_mr[1]]

            def attentionLayer(R, input_41ep_out):
                def forEverySentence(item):
                    temp = T.dot(item, WForAT)
                    # ???? change this
                    re = T.dot(temp, R)
                    return re

                # slist就是ei
                slist, noup = theano.scan(forEverySentence, sequences=input_41ep_out)

                aForRj = T.nnet.softmax(slist)[0]

                def mulWeight(sentence, weight):
                    return sentence * weight

                # 句子数×(690+120)
                newSRep, noup = theano.scan(mulWeight, sequences=[input_41ep_out, aForRj])

                # 1×(690+120)
                finalresult = T.sum(newSRep, axis=0)

                return finalresult

            # AT层数×1×(690+120)
            newSRepAf, _ = theano.scan(attentionLayer, outputs_info=RForAT[idx],
                                       non_sequences=input_41ep, n_steps=NUMBER)

            # finalresult = T.sum(newSRepAf[-1], axis=0)

            # return finalresult
            # 一次做完吧

            return newSRepAf[-1]

        # (50, (690+120))
        my_sec_add_out, _ = theano.scan(fn=forEveryExample, sequences=ep2m, non_sequences=[cp_out])

        return my_sec_add_out

    idx = T.ivector()
    # ok = (26,50,(690+120))
    ok, up = theano.scan(forEveryRelation, sequences=[idx],
                         non_sequences=[ep2m, new_cp_out])

    # (26, 50, (690 + 120))
    normalre, dropoutre = my_mlp_layer.feed(idx, ok,
                                            input_shape=(_N_RELATIONS, _BATCH_SIZE, (_MLP_SHAPE[0] + 2 * _IMG_W)))

    # input_shape = (batch_size = 50,_MLP_SHAPE[0] = 690)
    # normalre = 26 *50 *2
    # normalre, dropoutre = my_mlp_layer.feed(otheridx, my_sec_add_out, input_shape=(_BATCH_SIZE, _MLP_SHAPE[0]))

    # return [normalre, dropoutre]
    # return [normalre, dropoutre, my_sec_add_out]

    # idx = T.ivector()
    # otheridx = T.ivector()

    # ep2m = T.imatrix()
    # cp_out = T.matrix()

    #

    # [normalre, dropoutre, my], up = theano.scan(forEveryRelation, sequences=[idx],
    #                                             non_sequences=[otheridx, ep2m, cp_out])

    fiNormalre = T.transpose(normalre)
    fiDropoutre = T.transpose(dropoutre)

    # 用这个来算cost
    # 例子数×26
    # dropout_score_batch = mlp_layer.dropout_layers[-1].score
    # dropout_p_ygx_batch = T.nnet.sigmoid(dropout_score_batch)

    dropout_p_ygx_batch = fiDropoutre

    # 用这个来预测
    # score_batch = mlp_layer.layers[-1].score
    # p_ygx_batch = T.nnet.sigmoid(score_batch)

    p_ygx_batch = fiNormalre

    obz_lr_masks = my_mlp_layer.lrmask

    # 例子数×26维的矩阵，里面的数大于0.5就变成1，不大于0.5就变成0
    predictions = predict_relations(p_ygx_batch)

    pred_pscores = p_ygx_batch

    logging.info(' - Cost, params and grads ...')

    # 用这个更新权重
    # 计算损失的时候没有用到前面的score，score就是乘出来的26维的向量
    # 第二个参数是把score用sigmoid做归一化得出来的结果
    dropout_cost = compute_cost(dropout_p_ygx_batch, ys)

    cost = compute_cost(p_ygx_batch, ys)

    op_params = []
    params = []
    op_params += [Wordv]
    # is_static = False
    if not w2v_static:  # if word vectors are allowed to change, add them as model hyper_parameters
        params += [Wordv]

    op_params += [PF1v, PF2v]
    params += [PF1v, PF2v]

    op_params += cp_params
    params += cp_params

    op_params += my_mlp_layer.params
    params += my_mlp_layer.params

    op_params += [WForAT, RForAT, WForATData, linearW]
    params += [WForAT, RForAT, WForATData, linearW]

    # op_params += [WForAT, RForAT]
    # params += [WForAT, RForAT]

    logging.info('Params to update: ' + str(', '.join([param.name for param in params])))
    logging.info('Params to output: ' + str(', '.join([op_param.name for op_param in op_params])))

    # 5. 权重更新方式.
    grad_updates = lasagne.updates.adadelta(dropout_cost, params)

    # 6. 定义theano_function
    train_x_m, train_PFinfo_m, train_ep2m, train_y, test_x_m, test_PFinfo_m, test_ep2m, test_y = datasets

    logging.info('Compiling train_update_model...')
    output_list = [cost, dropout_cost]
    if conf.getboolean('mode', 'output_details'):
        output_list += ([obz_lr_masks, p_ygx_batch] + op_params)
    train_update_model = theano.function([bch_idx], output_list, updates=grad_updates,
                                         name='train_update_model',
                                         givens={
                                             xs: get_1batch_x_m(bch_idx, train_x_m, train_ep2m),
                                             pfinfos: get_1batch_x_m(bch_idx, train_PFinfo_m, train_ep2m),
                                             ep2m_raw: train_ep2m[bch_idx * _BATCH_SIZE: (bch_idx + 1) * _BATCH_SIZE],
                                             ys: train_y[bch_idx * _BATCH_SIZE: (bch_idx + 1) * _BATCH_SIZE],
                                             idx: np.arange(26, dtype='int32'),
                                         }, )
    #  }, on_unused_input='warn')

    logging.info('Compiling set_zero function ...')
    Wordv_0sline = T.vector("Wordv_0sline", dtype=theano.config.floatX)
    PFv_0sline = T.vector("PFv_0sline", dtype=theano.config.floatX)
    set_zero = theano.function([Wordv_0sline, PFv_0sline], updates=[
        (Wordv, T.set_subtensor(Wordv[0, :], Wordv_0sline)),
        (PF1v, T.set_subtensor(PF1v[0, :], PFv_0sline)),
        (PF2v, T.set_subtensor(PF2v[0, :], PFv_0sline))
    ])

    logging.info('Compiling trainset_error_model ...')
    trainset_error_model = theano.function([bch_idx], [predictions, pred_pscores], givens={
        xs: get_1batch_x_m(bch_idx, train_x_m, train_ep2m),
        pfinfos: get_1batch_x_m(bch_idx, train_PFinfo_m, train_ep2m),
        ep2m_raw: train_ep2m[bch_idx * _BATCH_SIZE: (bch_idx + 1) * _BATCH_SIZE],
        idx: np.arange(26, dtype='int32'),
    })

    logging.info('Compiling testset_error_model ...')
    testset_error_model = theano.function([bch_idx], [predictions, pred_pscores], givens={
        xs: get_1batch_x_m(bch_idx, test_x_m, test_ep2m),
        pfinfos: get_1batch_x_m(bch_idx, test_PFinfo_m, test_ep2m),
        ep2m_raw: test_ep2m[bch_idx * _BATCH_SIZE: (bch_idx + 1) * _BATCH_SIZE],
        idx: np.arange(26, dtype='int32'),
    })

    init_LR_W = my_mlp_layer.dropout_layers[-1].W.get_value()
    init_LR_b = my_mlp_layer.dropout_layers[-1].b.get_value()

    return train_update_model, trainset_error_model, testset_error_model, set_zero, init_LR_W, init_LR_b


def unzip_ep_info(dataset_x):
    """
    将数据集从以ep为中心转换成以mention为中心.
    :param dataset_x: 组织形式为'ep-mentions'的数据集.
    :return: 数据集中的mentions列表, 以及'ep-mention_range'列表.
    """

    dataset_m_li = list()
    ep2mr_li = list()
    PFinfo_m_li = list()
    start_idx = 0
    for ep in dataset_x:
        # ms为padded_idx_sent的列表,m_pfinfo为PFinfo的列表
        ms, m_pfinfo = zip(*ep)

        dataset_m_li.extend(ms)
        PFinfo_m_li.extend(m_pfinfo)

        mentions_range = (start_idx, start_idx + len(ms))
        ep2mr_li.append(mentions_range)
        start_idx += len(ms)

    return dataset_m_li, PFinfo_m_li, ep2mr_li


def get_1batch_x_m(batch_idx, input_mentions, ep2mlist):
    start_pos = ep2mlist[batch_idx * _BATCH_SIZE][0]
    end_pos = ep2mlist[(batch_idx + 1) * _BATCH_SIZE - 1][1]
    return input_mentions[start_pos: end_pos]


def run_mimlcnnre(file_name):
    logging.info('Running: {0} ...'.format(sys.argv[0]))

    logging.info("Loading data ...")
    x = cPickle.load(open(file_name, "rb"))
    datasets, Wordv_init, word2id, vocab, dataset_status = x[0], x[1], x[2], x[3], x[4]

    # raw_train是列表，每个列表里面都是[88维的向量+句子信息]这样的二元列表
    raw_train_x = datasets[0]
    raw_train_y = datasets[1]
    raw_test_x = datasets[2]
    raw_test_y = datasets[3]

    pos_train_y = filter(lambda x: x[0] != 1, raw_train_y)

    logging.info('  - input trainset has {0} entity pairs, '.format(len(raw_train_x)))
    logging.info("  - input trainset has {0} positive entity pairs".format(len(pos_train_y)))
    logging.info("  - input trainset has {0} triples".format(np.sum(np.array(pos_train_y))))

    pos_test_y = filter(lambda x: x[0] != 1, raw_test_y)

    logging.info('  - input testset has {0} entity pairs, '.format(len(raw_test_x)))
    logging.info("  - input testset has {0} positive entity pairs".format(len(pos_test_y)))
    logging.info("  - input testset has {0} triples".format(np.sum(np.array(pos_test_y))))

    # 88维
    img_h = len(raw_train_x[0][0][0])

    # print img_h

    logging.info("[!] Converting data format into NA_AS_0s")

    # 去除掉26维关系向量中的第一位
    raw_train_y = [ep_y[1:] for ep_y in raw_train_y]
    raw_test_y = [ep_y[1:] for ep_y in raw_test_y]

    # 断言为26维
    assert _N_RELATIONS == len(raw_train_y[0])

    # 1. 数据集处理: 补齐到batch_size的整数倍
    logging.info('Processing datasets (on entity pair level) ...')
    np.random.seed(3435)
    logging.info('  - Processing trainset')
    raw_train_idx = range(len(raw_train_x))
    raw_train_ixy = zip(raw_train_idx, raw_train_x, raw_train_y)

    if len(raw_train_x) % _BATCH_SIZE != 0:
        n_extra_train = _BATCH_SIZE - len(raw_train_x) % _BATCH_SIZE
        logging.info('    - Aligning: n_trainset % batch_size !=0, sample {0} eps as supplements'.format(n_extra_train))

        # 打乱下标序列，随机取前2个，(假设缺2个)
        extra_train_idx = np.random.permutation(raw_train_idx)[:n_extra_train]
        extra_train_ixy = [raw_train_ixy[i] for i in extra_train_idx]

        aligned_train_ixy = raw_train_ixy + extra_train_ixy
    else:
        aligned_train_ixy = raw_train_ixy

    # 补齐后的训练数据肯定是_BATCH_SIZE倍数
    assert len(aligned_train_ixy) % _BATCH_SIZE == 0

    # 随机打乱训练数据
    permutated_train_ixy = [aligned_train_ixy[i] for i in np.random.permutation(range(len(aligned_train_ixy)))]
    # 必定能整除
    n_train_batch = len(permutated_train_ixy) / _BATCH_SIZE
    # 取全部，这有啥意义吗
    train_ixy = permutated_train_ixy[: n_train_batch * _BATCH_SIZE]
    train_idx, train_x, train_y = zip(*train_ixy)

    logging.info('  - Processing testset')
    raw_test_idx = range(len(raw_test_x))
    raw_test_ixy = zip(raw_test_idx, raw_test_x, raw_test_y)
    if len(raw_test_x) % _BATCH_SIZE != 0:
        n_extra_test = _BATCH_SIZE - len(raw_test_x) % _BATCH_SIZE
        logging.info('    - Aligning: n_testset % batch_size !=0, sample {0} eps as supplements'.format(n_extra_test))
        extra_test_ixy = raw_test_ixy[:n_extra_test]
        aligned_test_ixy = raw_test_ixy + extra_test_ixy
    else:
        aligned_test_ixy = raw_test_ixy
    assert len(aligned_test_ixy) % _BATCH_SIZE == 0

    test_idx, test_x, test_y = zip(*aligned_test_ixy)
    n_test_bch = len(aligned_test_ixy) / _BATCH_SIZE
    # 数据处理完毕
    logging.info(' - n_trainset_batches, n_test_batches = {0}, {1}'.format(n_train_batch, n_test_bch))

    # 2. Check变换后的数据集
    logging.info('Checking data ...')
    assert set(train_idx) == set(raw_train_idx)
    assert set(test_idx) == set(raw_test_idx)

    # train_idx也被打乱了
    for i, x, y in zip(train_idx, train_x, train_y):
        assert raw_train_x[i] == x and raw_train_y[i] == y
    for i, x, y in zip(test_idx, test_x, test_y):
        assert raw_test_x[i] == x and raw_test_y[i] == y
    logging.info('  - Pass')

    # 3. ep转换为mention
    logging.info('Converting EP to mention and range...')
    train_x_m, train_PFinfo_m, train_ep2mr = unzip_ep_info(train_x)
    test_x_m, test_PFinfo_m, test_ep2mr = unzip_ep_info(test_x)

    #  4. 将数据集, PF特征, 以及输入参数转换为shared
    logging.info('Placing datasets into theano shared variables...')
    train_x_m_srd, train_PFinfo_m_srd, train_ep2m_srd, train_y_srd = shared_dataset(train_x_m, train_PFinfo_m,
                                                                                    train_ep2mr, train_y)
    test_x_m_srd, test_PFinfo_m_srd, test_ep2m_srd, test_y_srd = shared_dataset(test_x_m, test_PFinfo_m, test_ep2mr,
                                                                                test_y)
    datasets_ready = (
        train_x_m_srd, train_PFinfo_m_srd, train_ep2m_srd, train_y_srd,
        test_x_m_srd, test_PFinfo_m_srd, test_ep2m_srd, test_y_srd
    )

    logging.info('Placing Wordv, PF1v, PF2v into theano shared variables...')
    PF1_raw = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]), dtype=theano.config.floatX)
    padPF1 = np.zeros((1, 5), dtype=theano.config.floatX)  # 对应padding

    # 102×5的矩阵
    # 第0行是全0
    # 之后剩下的101行全部是-1~1的平均分布
    PF1_init = np.vstack((padPF1, PF1_raw))

    PF2_raw = np.asarray(rng.uniform(low=-1, high=1, size=[101, 5]), dtype=theano.config.floatX)
    padPF2 = np.zeros((1, 5), dtype=theano.config.floatX)

    # 102×5的矩阵
    # 第0行是全0
    # 之后剩下的101行全部是-1~1的平均分布
    PF2_init = np.vstack((padPF2, PF2_raw))

    # ******************
    # attention

    my_in = (690 + 2 * _IMG_W)
    my_out = (690 + 2 * _IMG_W)
    W_bound = np.sqrt(6. / (my_in + my_out))

    WForAT_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[my_in, my_out]),
                             dtype=theano.config.floatX)

    RForAT_init = np.asarray(rng.uniform(low=1, high=1, size=[26, my_out]), dtype=theano.config.floatX)
    # 已验证果然只有R[0]更新
    # RForAT_a = np.asarray(rng.uniform(low=-1, high=1, size=[1, 200]), dtype=theano.config.floatX)
    # RForAT_b = np.zeros((25, 200), dtype=theano.config.floatX)
    # RForAT_init = np.vstack((RForAT_a, RForAT_b))
    # *****************
    data_my_in = _IMG_W * 3
    W_bound = np.sqrt(1. / (data_my_in))

    WForATData_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[data_my_in]),
                                 dtype=theano.config.floatX)

    linearW_init = np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=[_IMG_W, _IMG_W * 2]),
                              dtype=theano.config.floatX)

    Wordv = theano.shared(value=Wordv_init, name="Wordv")

    PF1v = theano.shared(value=PF1_init, name="PF1v")
    PF2v = theano.shared(value=PF2_init, name="PF2v")

    # ******************
    #
    WForAT = theano.shared(value=WForAT_init, name="WForAT")
    RForAT = theano.shared(value=RForAT_init, name="RForAT")
    WForATData = theano.shared(value=WForATData_init, name="WForATData")
    linearW = theano.shared(value=linearW_init, name="linearW")
    # *****************

    assert Wordv.dtype == PF1v.dtype == PF2v.dtype == theano.config.floatX

    # 5. 定义模型结构
    logging.info('Defining model structure ...')
    # 源代码
    # train_update_model, trainset_error_model, testset_error_model, set_zero, init_LR_W, init_LR_b = \
    #     model_mimlcnn(datasets_ready, Wordv, PF1v, PF2v, img_h)

    train_update_model, trainset_error_model, testset_error_model, set_zero, init_LR_W, init_LR_b = \
        model_mimlcnn(datasets_ready, Wordv, PF1v, PF2v, img_h, RForAT, WForAT, WForATData, linearW)

    # 6. 训练与验证效果
    logging.info('Training starts...')
    epoch = 1
    while epoch <= _N_EPOCHS:
        logging.info('Epoch: ' + str(epoch))
        index_seq_to_train = np.random.permutation(range(n_train_batch)) if _SHUFFLE_BATCH else range(n_train_batch)
        for batch_index in index_seq_to_train:
            if not conf.getboolean('mode', 'output_details'):
                cost, dropout_cost= train_update_model(batch_index)
            else:
                # LR_W_value1, LR_b_value1, LR_W_value2, LR_b_value2, LR_W_value3, LR_b_value3, \
                # LR_W_value4, LR_b_value4, LR_W_value5, LR_b_value5, LR_W_value6, LR_b_value6, \
                # LR_W_value7, LR_b_value7, LR_W_value8, LR_b_value8, LR_W_value9, LR_b_value9, \
                # LR_W_value10, LR_b_value10, LR_W_value11, LR_b_value11, LR_W_value12, LR_b_value12, \
                # LR_W_value13, LR_b_value13, LR_W_value14, LR_b_value14, LR_W_value15, LR_b_value15, \
                # LR_W_value16, LR_b_value16, LR_W_value17, LR_b_value17, LR_W_value18, LR_b_value18, \
                # LR_W_value19, LR_b_value19, LR_W_value20, LR_b_value20, LR_W_value21, LR_b_value21, \
                # LR_W_value22, LR_b_value22, LR_W_value23, LR_b_value23, LR_W_value24, LR_b_value24, \
                # LR_W_value25, LR_b_value25, LR_W_value26, LR_b_value26, \

                # LR_W_value1, LR_b_value1, \

                cost, dropout_cost, lr_masks, p_ygx_mention, Wordv_value, PF1v_value, PF2v_value, \
                CP_W_value, CP_b_value, LR_W_value, LR_b_value, \
                WForAT_value, RForAT_value = train_update_model(
                    batch_index)  # observe
                obz_dict = {
                    'Wordv': Wordv_value,
                    'PF1v': PF1v_value,
                    'PF2v': PF2v_value,
                    'CP_W': CP_W_value,
                    'CP_b': CP_b_value,
                    'LR_W': LR_W_value,
                    'LR_b': LR_b_value,
                    'WForAT': WForAT_value,
                    'RForAT': RForAT_value,
                    'lr_mask': lr_masks,
                    "p_ygx_mention": p_ygx_mention,
                }

                if batch_index % 250 == 0:
                    logging.info('    - outputing param details, epoch:{0}, batch:{1}'.format(epoch, batch_index))
                    save_observations(batch_index, epoch, obz_dict)

            if batch_index % 50 == 0:
                logging.info(
                    ' - minibatch_idx: {0}, cost={1}, dropout_cost={2}'.format(batch_index, cost, dropout_cost))
            Wordv_0sline = np.zeros(shape=Wordv_init.shape[1], dtype=theano.config.floatX)
            PFv_0sline = np.zeros(shape=PF1_init.shape[1], dtype=theano.config.floatX)
            set_zero(Wordv_0sline, PFv_0sline)
        logging.info("Evaluating...")
        evaluate_performance(epoch, trainset_error_model, n_train_batch, testset_error_model, n_test_bch, raw_test_y,
                             test_y)
        epoch += 1


def save_observations(minibatch_index, epoch, obz_dict):
    for (obz_name, obz_value) in obz_dict.items():
        l2sum = np.sum(np.square(obz_value))
        if np.isnan(l2sum):
            logging.warning('[WARNING] param {0} get nan. value={1}'.format(obz_name, obz_value))
        logging.info('      - saving {0}, l2sum={1}, shape={2}'.format(obz_name, l2sum, obz_value.shape))
        filename = '{0}/{1}_batch{2}_epoch{3}'.format(_OBSERVATION_DIR, obz_name, minibatch_index, epoch)
        if obz_name == 'Wordv':
            logging.info('          - printing the entire Wordv takes forever. print Wordv[:20000] instead.')
            np.savetxt(filename, obz_value[:2000], fmt='%.8e', delimiter='\t\t')
        else:
            with open(filename, 'w') as f:
                if obz_name == 'conv_out':
                    assert obz_value.ndim == 3
                    for i in range(obz_value.shape[0]):
                        np.savetxt(f, obz_value[i], fmt='%+.8f', delimiter='\t\t', header="# mention:{0}".format(i),
                                   comments='\n')
                else:
                    f.write(str(obz_value))


def evaluate_performance(epoch, trainset_error_model, n_trainset_batches, testset_error_model, n_testset_batches,
                         raw_testset_y, aligned_testset_y, save_all_prs=True):
    testset_results = [testset_error_model(batch_idx) for batch_idx in range(n_testset_batches)]
    testset_predictions_batches, testset_pscores_batches = zip(*testset_results)

    # 全都恢复成原始长度的ndarray.
    n_last_test_batch = len(raw_testset_y) % _BATCH_SIZE
    if n_last_test_batch == 0:
        n_last_test_batch = _BATCH_SIZE
    assert (len(testset_pscores_batches) - 1) * _BATCH_SIZE + n_last_test_batch == len(raw_testset_y)

    testset_pscores = np.concatenate(
        [np.concatenate(list(testset_pscores_batches[:-1])), testset_pscores_batches[-1][0:n_last_test_batch]])
    testset_predictions = np.row_stack(
        [np.row_stack(list(testset_predictions_batches[:-1])), testset_predictions_batches[-1][0:n_last_test_batch]])
    testset_goldens = np.asarray(raw_testset_y)
    for epi in range(len(testset_goldens)):
        assert (testset_goldens[epi] == aligned_testset_y[epi]).all()

    if save_all_prs:
        logging.info('  - saving all model output on testset')
        save_prs_and_labels(epoch, testset_pscores, testset_predictions, testset_goldens)

    predicted_triples = predict_triples(epoch, testset_pscores, testset_predictions, testset_goldens)
    cal_prs_and_plot_curve(epoch, predicted_triples, testset_goldens)


def predict_triples(epoch, scores, y_preds, y_goldens):
    predicted_triples = []
    # 1, 挑出triples(NA不为0)
    for ep_idx in range(y_preds.shape[0]):
        for label_idx in range(y_preds.shape[1]):
            if y_preds[ep_idx][label_idx] == 1:
                predicted_triples.append((ep_idx, label_idx, float(scores[ep_idx][label_idx])))

    # 2. 将这些triples按score排序
    predicted_triples.sort(key=lambda x: -x[2])

    # 3. 存储按score降序排序好的predicted triples和pr_value.
    logging.info("  - saving predicted triples")
    with open("{0}/triples_e{1}.txt".format(_PRCURVE_DATA_DIR, epoch), 'w') as op_file:
        op_file.write(OP_ATT_SEP.join(['ep_idx', 'predicted_label', 'golden_labels', 'triple_score']) + '\n')
        for triple in predicted_triples:
            indices_gold = np.nonzero(y_goldens[triple[0]])[0]
            str_indices_gold = '_'.join(str(ele) for ele in indices_gold)
            line = OP_ATT_SEP.join([str(triple[0]), str(triple[1]), str_indices_gold, str(triple[2])]) + '\n'
            op_file.write(line)
    return predicted_triples


def cal_prs_and_plot_curve(epoch, predicted_triples, y_goldens):
    # 1. 根据预测和真实值算precisions, recalls.
    recalls = [0.0]
    precisions = [0.0]
    tp = 0.0
    tp_fp = 0.0
    total = np.sum(y_goldens[:, 1:])
    for triple in predicted_triples:
        tp_fp += 1
        if y_goldens[triple[0], triple[1]] == 1:
            tp += 1
        precision = tp / tp_fp
        recall = tp / float(total)
        if precision != precisions[-1] or recall != recalls[-1]:
            precisions.append(precision)
            recalls.append(recall)
    logging.info('  - tp={0}, tp+fp={1}, total={2}'.format(tp, tp_fp, total))
    precisions = precisions[1:]
    recalls = recalls[1:]
    auc_value = auc(recalls, precisions)
    with open("{0}/prs_Epoch_{1}.txt".format(_PRCURVE_DATA_DIR, epoch), 'w') as op_file:
        for p, r in zip(precisions, recalls):
            op_file.write(" ".join([str(p), str(r)]) + '\n')

    # 划分
    split_recall = 0.3
    fig_dpi = 200
    f_precisions = []
    f_recalls = []
    l_precisions = []
    l_recalls = []
    for p, r in zip(precisions, recalls):
        if r < split_recall:
            f_precisions.append(p)
            f_recalls.append(r)
        else:
            l_precisions.append(p)
            l_recalls.append(r)
    f_auc = 0.0
    l_auc = 0.0
    if len(f_precisions) >= 2:
        f_auc = auc(f_recalls, f_precisions)
    if len(l_precisions) >= 2:
        l_auc = auc(l_recalls, l_precisions)

    op_filename = _PRCURVE_DATA_DIR + "/auc.txt"
    if not os.path.exists(op_filename):
        op_file = open(op_filename, 'w')
        op_file.write(OP_ATT_SEP.join(['epoch', 'former_auc', 'latter_auc', 'all_auc']) + "\n")
    else:
        op_file = open(op_filename, 'a')
    op_file.write(OP_ATT_SEP.join([str(epoch), str(f_auc), str(l_auc), str(auc_value)]) + "\n")
    op_file.close()

    # 3. 根据precisions, recalls画图.
    plt.clf()
    plt.plot(recalls, precisions, label='PRC')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.1, 1.01])
    plt.xlim([0.0, 0.6])
    plt.grid(True)
    plt.title('PRC@e{0}, auc={0}'.format(epoch, auc_value))
    plt.legend(loc="upper right")
    fig_filename = "{0}/e{1}_auc_{2:.6f}_{3:.6f}.png".format(_PRCURVE_DATA_DIR, epoch, f_auc, l_auc)
    plt.savefig(fig_filename, dpi=fig_dpi)


def save_prs_and_labels(epoch, pscore_all, y_pred_all, testset_y_all):
    filename = "{0}/all_preds_e{1}.txt".format(_RESULT_DATA_DIR, epoch)
    op_file = open(filename, 'wb')
    op_file.write(OP_ATT_SEP.join(['ep_idx', 'pred', 'gold', 'score_on_pred_label']) + '\n')
    for i in range(len(pscore_all)):
        if i % _BATCH_SIZE == 0:
            op_file.write('\n')
            op_file.write('Batch: {0}'.format(i / _BATCH_SIZE) + '\n')
        indices_pred_ep = np.nonzero(y_pred_all[i])[0]
        indices_gold_ep = np.nonzero(testset_y_all[i])[0]
        str_indices_pred = '_'.join(str(ele) for ele in indices_pred_ep)
        str_indices_gold = '_'.join(str(ele) for ele in indices_gold_ep)
        line = OP_ATT_SEP.join([str(i), str_indices_pred, str_indices_gold, str(pscore_all[i])]) + '\n'
        op_file.write(line)
    op_file.close()


def shared_dataset(dataset_x_m, PF_m, ep2m_list, dataset_y, borrow=True):
    """  Function that loads the dataset into shared variables.
    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x_m = theano.shared(np.asarray(dataset_x_m, dtype=theano.config.floatX))
    shared_PF_m = theano.shared(np.asarray(PF_m, dtype=theano.config.floatX))
    shared_ep2m = theano.shared(np.asarray(ep2m_list, dtype=theano.config.floatX))
    shared_y = theano.shared(np.asarray(dataset_y, dtype=theano.config.floatX))
    return T.cast(shared_x_m, 'int32'), T.cast(shared_PF_m, 'int32'), T.cast(shared_ep2m, 'int32'), T.cast(shared_y,
                                                                                                           'int32')


def my_sgd_updates(cost, params, Words):
    learning_rate = conf.getfloat('hpyer_parameters', 'learning_rate')
    updates = OrderedDict({})
    for param in params:
        if param.name == 'Words_batch':
            updates[Words] = T.inc_subtensor(param, -learning_rate * T.grad(cost, param))
        else:
            updates[param] = param - learning_rate * T.grad(cost, param)
    return updates


def kim13_adadelta_with_part_maxnorm(cost, params, rho=0.95, epsilon=1e-6, norm_lim=9, no_norm=('Wordv', 'PFv')):
    """ adadelta update rule, mostly from https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta) """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step = -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name not in no_norm):
            logging.info('col max-norming {0}'.format(param.name))
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param
    return updates


def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)
    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)


def get_record_dir():
    ts = datetime.datetime.now().strftime("%m%d.%H%M%S")
    rcd_dir = os.path.split(_DATASETS_READY_DIR)[0] + '/Records_' + ts
    return rcd_dir


def create_record_folders(record_dir):
    global _DATASETS_READY_DIR, _RESULT_DATA_DIR, _PRCURVE_DATA_DIR, _OBSERVATION_DIR
    os.mkdir(record_dir)
    _DATASETS_READY_DIR = record_dir + '/' + os.path.split(_DATASETS_READY_DIR)[1]
    _RESULT_DATA_DIR = record_dir + '/' + os.path.split(_RESULT_DATA_DIR)[1]
    _PRCURVE_DATA_DIR = record_dir + '/' + os.path.split(_PRCURVE_DATA_DIR)[1]
    _OBSERVATION_DIR = record_dir + '/' + os.path.split(_OBSERVATION_DIR)[1]
    os.mkdir(_DATASETS_READY_DIR)
    os.mkdir(_RESULT_DATA_DIR)
    os.mkdir(_PRCURVE_DATA_DIR)
    os.mkdir(_OBSERVATION_DIR)


def save_environment(record_dir):
    cur_dir = sys.path[0]
    for filename in os.listdir(cur_dir):
        if filename.endswith('.py') or filename.endswith('.ini'):
            shutil.copyfile(filename, record_dir + '/' + filename)


def configure_logger(dirname):
    log_format = '%(asctime)s [%(levelname)s from line:%(lineno)d] %(message)s'
    log_datefmt = conf.get('logging', 'datefmt')
    logging.basicConfig(level=logging.DEBUG, format=log_format, datefmt=log_datefmt,
                        filename=dirname + '/result', filemode='w')

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format, log_datefmt))
    logging.getLogger('').addHandler(console)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='datasets file path generated by process_data.py', type=str)
    parser.add_argument('-c', help='comments of this run. It will be stored in reminders.txt', type=str)
    parser.add_argument('-t', help='string tagged on cur dir name', type=str)
    parser.add_argument('-o', help='output dir name, stored in ./data/', type=str)
    parsed = parser.parse_args()
    return vars(parsed)


if __name__ == "__main__":
    args = parse_args()
    if not args['i']:
        print "[Error] You need to specify a input file (.p) ."
        # exit()
        input_dataset_path = "test.p"
    else:
        input_dataset_path = args['i']

    # input_dataset_path = args['i']

    dir_surfix = args['t'] or ""

    if args["o"]:
        cur_record_dir = os.path.split(_DATASETS_READY_DIR)[0] + "/" + args["o"]
    else:
        cur_record_dir = get_record_dir() + "_" + dir_surfix
    create_record_folders(cur_record_dir)
    save_environment(cur_record_dir)
    configure_logger(cur_record_dir)
    logging.info('Records will be stored in dir: {0}'.format(cur_record_dir))

    if args['c']:
        logging.info('Comments are writing to reminders.txt')
        with open(cur_record_dir + "/reminders.txt", 'w') as reminder_file:
            reminder_file.write(args['c'] + "\n")

    logging.info('Input datasets file: {0}'.format(input_dataset_path))

    run_mimlcnnre(input_dataset_path)
