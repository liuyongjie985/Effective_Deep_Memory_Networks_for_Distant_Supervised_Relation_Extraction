# -*- coding: UTF-8 -*-

import cPickle
import re
import os
import tarfile
import argparse
import numpy as np

rng = np.random.RandomState(3435)
import theano
from MIMLEntityPair import MIMLEntityPair
import logging
import ConfigParser
from gensim.models.word2vec import Word2Vec
import collections

ini_filename = 'ModelControl.ini'
conf = ConfigParser.ConfigParser()
conf.read(ini_filename)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s from line:%(lineno)d] %(message)s',
    datefmt=conf.get('logging', 'datefmt'),
    # filename = conf.get('logging','filepath')+'\mimlrecnn.log',
    # filemode = conf.get('logging','filemode')
)
theano.config.floatX = conf.get('theano_setup', 'floatX')


def read_from_file(filename, ds_split=None, read_feature=False, merge_ep=False):
    """
    read eps from file.
    :param filename: .txt file to read
    :param ds_split: in {'train', 'test'}. default=None
    :param read_feature: whether or not read features.
    :param merge_ep: whether or not merge ep (MISL->MIML)
    :return: ep list in the file.
    """
    f = open(filename, 'rb')
    ep_list = list()
    lines_collect = list()
    line = f.readline()
    print 'set merge_ep = {0}'.format(merge_ep)
    while line:
        if line.startswith(MIMLEntityPair.ER_SEP):  # 如果是'------', 就说明上一个ep读完了.
            ep = MIMLEntityPair(lines_collect, ds_split, read_feature)
            if merge_ep and not ('NA' in ep.relations):
                merge = False
                for ep_in_list in ep_list:
                    if ep.has_same_ep_name(ep_in_list):
                        merge = True
                        ep_in_list_sents = set([m_info['sentence'] for m_info in ep_in_list.mentions_info])
                        ep_sents = set([m_info['sentence'] for m_info in ep.mentions_info])
                        if ep_in_list_sents != ep_sents:
                            msg = "The eps to be merged have different mentions: " + str(
                                ep_in_list_sents) + '  <-  ' + str(ep_sents)
                            raise NotImplementedError, msg
                        else:  # merge
                            ep_in_list.relations.extend(ep.relations)
                            ep_in_list.rels_id.extend(ep.rels_id)
                        MIMLEntityPair.EPCOUNT_ALL -= 1
                        MIMLEntityPair.MCOUNT_ALL -= len(ep.mentions_info)
                        break
                if not merge:
                    ep_list.append(ep)
            else:
                ep_list.append(ep)
            lines_collect = list()
        else:
            lines_collect.append(line)
        line = f.readline()
    f.close()

    n_more = 0
    for ep in ep_list:
        if len(ep.relations) > 1:
            n_more += 1
    print 'File \'{0}\' is read. n_ep: {1}, ep with more than one relation: {2}'.format(filename, len(ep_list), n_more)
    return ep_list


def build_data(raw_eps, clean_string=False):
    revs = []
    vocab = collections.defaultdict(int)
    for raw_ep in raw_eps:
        PFinfo = list()
        sentences_ready = list()
        e1n_con = ("$$" + "##".join(raw_ep.e1_name.split()) + "$$")
        e2n_con = ("$$" + "##".join(raw_ep.e2_name.split()) + "$$")

        e1n_tokens = raw_ep.e1_name.split()
        e2n_tokens = raw_ep.e2_name.split()

        for m_info in raw_ep.mentions_info:
            if len(m_info["sentence"].strip().split()) > conf.getint('process_dataset', 'truncate_sent_length'):
                print "[Dropping long mention] [ep: {0}, ({1}, {2}), ds_split={3}] Len={4}, Sent= {5}".format(
                    raw_ep.ep_sn, raw_ep.e1_name, raw_ep.e2_name, raw_ep.ds_split,
                    len(m_info["sentence"].strip().split()), m_info["sentence"].strip())
                continue

            sent_4_loc = "B_-2 B_-1 " + m_info["sentence"].strip() + " B_1 B_2"
            sent_tokens_4loc = sent_4_loc.split()
            feat_for_loc = m_info["feat_for_loc"].strip()
            f_inverse, f_fm_left, f_type_fm, f_POSs, f_type_sm, f_sm_right = feat_for_loc.split('|')
            assert f_inverse == "inverse_false" or f_inverse == "inverse_true"
            f_inverse = True if f_inverse == "inverse_true" else False
            fe_tokens, se_tokens = (e1n_tokens, e2n_tokens) if not f_inverse else (e2n_tokens, e1n_tokens)
            fm_left_tokens = f_fm_left.split()
            if len(fm_left_tokens) > 2:
                fm_left_tokens = fm_left_tokens[-2:]
            assert len(fm_left_tokens) == 2
            sm_right_tokens = f_sm_right.split()
            if len(sm_right_tokens) > 2:
                sm_right_tokens = sm_right_tokens[:2]
            assert len(sm_right_tokens) == 2
            f_POSs = f_POSs.split()
            n_pos_l1_to_r1 = len(fm_left_tokens) + len(fe_tokens) + len(f_POSs) + len(se_tokens)

            cdd_l1_idx = []
            for i in range(len(sent_tokens_4loc) - 1):
                if sent_tokens_4loc[i] == fm_left_tokens[0] and sent_tokens_4loc[i + 1] == fm_left_tokens[1]:
                    cdd_l1_idx.append(i)
            filtered_l1_idx = []
            for idx in cdd_l1_idx:
                if fe_tokens == sent_tokens_4loc[idx + 2: idx + 2 + len(fe_tokens)]:
                    filtered_l1_idx.append(idx)
            fm_idx = -1  # 这两个都是开始的token的idx
            sm_idx = -1
            for idx in filtered_l1_idx:
                expected_smr1_idx = idx + n_pos_l1_to_r1
                expected_smr2_idx = idx + n_pos_l1_to_r1 + 1
                if expected_smr2_idx >= len(sent_tokens_4loc):  # 要求最右侧也要在索引范围
                    continue
                for i in range(expected_smr1_idx, len(sent_tokens_4loc) - 1):
                    if sent_tokens_4loc[i] == sm_right_tokens[0] and sent_tokens_4loc[i + 1] == sm_right_tokens[1]:
                        accept = True
                        reversed_sm_tokens = se_tokens[::-1]
                        for j in range(len(reversed_sm_tokens)):
                            # if not reversed_sm_tokens[j].lower() == sent_tokens_4loc[i-1-j].lower():
                            if not reversed_sm_tokens[j] == sent_tokens_4loc[i - 1 - j]:
                                accept = False
                                break
                        if accept:  # 找到了第二个出现的mention
                            fm_idx = idx + 2
                            sm_idx = i - len(se_tokens)
                if sm_idx != -1:
                    break

            assert fm_idx != -1 and sm_idx != -1
            # print "found. e1={0}, e2={1}, feat={2}, sent={3}".format(e1n_tokens, e2n_tokens, feat_for_loc, sent_4_loc)

            fm_name = " ".join(sent_tokens_4loc[fm_idx: fm_idx + len(fe_tokens)])
            sm_name = " ".join(sent_tokens_4loc[sm_idx: sm_idx + len(se_tokens)])
            assert fm_name == " ".join(fe_tokens) and sm_name == " ".join(se_tokens)
            #           print "found 2. e1={0}, e2={1}, fm = {2}, sm={3}, feat={4}, sent={5}".format(e1n_tokens, e2n_tokens,
            #               sent_tokens_4loc[fm_idx: fm_idx + len(fm_tokens)], sent_tokens_4loc[sm_idx: sm_idx + len(sm_tokens)], feat_for_loc, sent_4_loc)

            #     if sent_tokens_4loc[expected_m2r1_idx] == m2_right_tokens[0] and sent_tokens_4loc[expected_m2r2_idx] == m2_right_tokens[1]:
            #         cand2_l1_idx.append(idx)
            # # assert len(cand2_l1_idx) > 0      # 一开始是设为==1. 但是重复出现的套路会导致得到多个符合的.
            #
            # valid_cnt = 0
            # for l1_idx in cand2_l1_idx:
            #     fm_start_idx = l1_idx + 2
            #     sm_start_idx = l1_idx + 2 + len(fm_tokens) + len(f_POSs)
            #     fm_name = " ".join(sent_tokens_4loc[fm_start_idx: fm_start_idx + len(fm_tokens)])
            #     sm_name = " ".join(sent_tokens_4loc[sm_start_idx: sm_start_idx + len(sm_tokens)])
            #     if not f_inverse:
            #         if not (fm_name.lower() == raw_ep.e1_name.strip().lower() and sm_name.lower() == raw_ep.e2_name.strip().lower()):
            #             continue
            #     else:
            #         if not (fm_name.lower() == raw_ep.e2_name.strip().lower() and sm_name.lower() == raw_ep.e1_name.strip().lower()):
            #             continue
            sec1 = " ".join(sent_tokens_4loc[2: fm_idx])  # 滤掉B_
            sec2 = " ".join(sent_tokens_4loc[fm_idx + len(fe_tokens): sm_idx])
            sec3 = " ".join(sent_tokens_4loc[sm_idx + len(se_tokens): -2])
            fm_rep = "$$" + "##".join(fm_name.split()) + "$$"
            sm_rep = "$$" + "##".join(sm_name.split()) + "$$"
            rec_sentence = " ".join([sec1, fm_rep, sec2, sm_rep, sec3])
            rec_sent_tokens = rec_sentence.split()

            # 计算PF
            fm_idx = rec_sent_tokens.index(fm_rep)
            sm_idx = rec_sent_tokens.index(sm_rep)
            e1i, e2i = (fm_idx, sm_idx) if not f_inverse else (sm_idx, fm_idx)
            # assert rec_sent_tokens[e1i].lower() == e1n_con and rec_sent_tokens[e2i].lower() == e2n_con
            assert rec_sent_tokens[e1i] == e1n_con and rec_sent_tokens[e2i] == e2n_con
            slen = len(rec_sent_tokens)
            PFinfo.append([slen, e1i, e2i])
            sentences_ready.append(rec_sentence)

            for word in set(rec_sent_tokens):
                # vocab[word.lower()] += 1        # 所以vocab中的词都是小写的.
                vocab[word] += 1  # 所以vocab中的词都是小写的.

        if len(sentences_ready) == 0:  # 如果ep下面没有mention, 或者因trunctuate导致ep下面没有mention, 就弃掉这个ep
            print '    - [Dropping EP: No mention] ep: {0}, rel_id: {1}, ds_split: {2}, entity_pair: ({3},{4})'.format(
                raw_ep.ep_sn, raw_ep.rels_id, raw_ep.ds_split, raw_ep.e1_name, raw_ep.e2_name)
            continue

        revs.append({"labels": raw_ep.rels_id,  # int
                     "ds_split": raw_ep.ds_split,  # 'train' / 'test'
                     "e1name": e1n_con,  # str
                     "e2name": e2n_con,  # str
                     # "sentences": [s.lower() for s in sentences_ready],        # 句子的list.
                     "sentences": [s for s in sentences_ready],  # 句子的list.
                     "PFinfo": PFinfo,  # (sentence_length, e1_index, e2_idx)        注意这个idx指的是在pad之前的句子中的idx.
                     })
    return revs, vocab


# """
# 1. PFP = position feature pair.
# 2. name大小写:
#     - 需要考虑大小写的例子: 句子中有Media. 但是万一句子中又有media, 就找不准确了. (但是这种情况目前还没找到)
#     - 需要不考虑大小写的例子: 实体名:[Grand Duchy]. 句子: exclusive right to trade within the realm of Muscovy , a grand duchy including Moscow .
# 目前的方案还是先忽略大小写(因为NYT10的特性是实体基本是PERSON, LOCATION, ORGANIZATION), 到时候如果有因此导致的错误, 再说.
# """
# def cal_PFs_for_sentences(e1name, e2name, sentences):
#     sent2PFPlist = dict()    # 一个句子对应若干个pfps_sent数组.
#     for s in sentences:
#         sent2PFPlist[s] = list()
#     for s in sent2PFPlist.keys():
#         tokens = s.split()
#         e1_indices = []        # e1的所有下标
#         e2_indices = []
#         for idx in range(len(tokens)):
#             if tokens[idx] == e1name:
#                 e1_indices.append(idx)
#             if tokens[idx] == e2name:
#                 e2_indices.append(idx)
#         for e1i in e1_indices:
#             for e2i in e2_indices:
#                 # 根据这组(e1i, e2i)来确定sentence中每个词的pfp, 之后如果想找e1和e2的idx, 只需要找为0的pf所对应的idx即可.
#                 # 后面称由这样一组(e1i, e2i)所确定的句子sent中单词的pfp列表为句子sent的一种PF指派.
#                 wordidx_sent = range(len(tokens))
#                 pf1s_sent = map(lambda i: i - e1i, wordidx_sent)
#                 pf2s_sent = map(lambda i: i - e2i, wordidx_sent)
#                 pfps_sent = zip(pf1s_sent, pf2s_sent)
#                 sent2PFPlist[s].append(pfps_sent)
#
#     # used_sentPF_dict用来装用过一次的指派. 只有当所有的指派被使用过了之后, 用过一次的指派可以被继续使用.
#     # 这样做是因为有很多句子因出现在不同文章中而在训练集中重复出现, 所以有可能这个句子只有一组指派, 但是句子在原文中重复很多次.
#     # 这种情况下, 目前的做法是:
#     # ① 先分配不同的指派. 然后如果分配完后还有剩余的句子, 就**随机指定**一个之前分配过的指派.
#     # ② 先分配不同的指派. 然后如果分配完后还有剩余的句子, **丢弃** (去掉else段).
#     used_sentPF_dict = dict()
#     for s in sent2PFPlist.keys():
#         used_sentPF_dict[s] = list()
#
#     result_PFs = []
#     for s in sentences:
#         if len(sent2PFPlist[s]) > 0:
#             alloc = sent2PFPlist[s].pop(0)
#             used_sentPF_dict[s].append(alloc)
#             result_PFs.append(alloc)
#         else:
#             if len(used_sentPF_dict[s]) == 0:
#                 print 1
#             idx = np.random.randint(0, len(used_sentPF_dict[s]))
#             rand_alloc = used_sentPF_dict[s][idx]
#             result_PFs.append(rand_alloc)
#         assert len(result_PFs[-1]) == len(s.split())
#
#     for s, li in sent2PFPlist.items():
#         if len(li) > 0:
#             li_str = "(" + ",".join([str(p) for p in li]) + ")"
#             print '[WARNING!] allocation {0} not used for e1={1}, e2={2}, sent={3}'.format(li_str, e1name, e2name, s)
#
#     return result_PFs


def build_Wordv(word2vec_dict, k):
    """ Get word matrix. W[i] is the vector for word indexed by i. """
    vocab_size = len(word2vec_dict)
    word2id_dict = dict()
    W = np.zeros(shape=(vocab_size + 1, k))
    W[0] = np.zeros(k)
    i = 1
    for word in word2vec_dict:
        # print type(word), ' | ', word
        W[i] = word2vec_dict[word]
        # print type(W[i]), " | ", W[i]
        word2id_dict[word] = i
        i += 1
    return W, word2id_dict


# def load_word2vec(filepath, vocab):
#     """ 读入word_vecs """
#     word_vecs = {}
#     with open(filepath, "rb") as f:
#         header = f.readline()
#         vocab_size, layer1_size = map(int, header.split())
#         binary_len = np.dtype(theano.config.floatX).itemsize * layer1_size    # 每个词向量的长度.
#         for line in xrange(vocab_size):    # 对于word2vec文件中的每一行
#             word = []
#             while True:
#                 ch = f.read(1)
#                 if ch == ' ':
#                     word = ''.join(word)
#                     break
#                 if ch != '\n':
#                     word.append(ch)
#             if word in vocab:    # 如果word2vec的单词在我们的词典中, 就存储其词向量, 否则略过其词向量, 看下一个单词.
#                word_vecs[word] = np.fromstring(f.read(binary_len), dtype=theano.config.floatX)
#             else:
#                 f.read(binary_len)
#     return word_vecs, layer1_size

def load_word2vec_from_gsmodel(gsmodel_path, vocab):
    """ 读入word_vecs """
    word_vecs = {}
    model = Word2Vec.load(gsmodel_path)
    for word in vocab:
        if word in model:
            word_vecs[word] = model[word]
    return word_vecs


def add_unknown_words(word_vecs, vocab, k, min=1):
    """
    将出现在vocab中, 且至少出现在min_df篇文章中, 但却不在word_vecs中的单词的词向量初始化为[-0.25, 0.25]的随机数.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


# def clean_str(string, TREC=False):
#     """
#     Tokenization/string cleaning for all datasets except for SST. Every dataset is lower cased except for TREC
#     原始NYT语料需要clean(生成w2v), 而数据集中的句子已经进行过了处理, 就不需要这个操作了.
#     """
#     string = re.sub(r"[^A-Za-z0-9(),\.!;:?\'\`]", " ", string)
#     string = re.sub(r"\'s", " \'s", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"n\'t", " n\'t", string)
#     string = re.sub(r"\'re", " \'re", string)
#     string = re.sub(r"\'d", " \'d", string)
#     string = re.sub(r"\'ll", " \'ll", string)
#     string = re.sub(r"\'ve", " \'ve", string)
#     string = re.sub(r"\'\'", " \'\' ", string)
#     string = re.sub(r",", " , ", string)
#     string = re.sub(r"\.", " . ", string)
#     string = re.sub(r"!", " ! ", string)
#     string = re.sub(r":", " : ", string)
#     string = re.sub(r";", " ; ", string)
#     string = re.sub(r"/", " / ", string)
#     string = re.sub(r"\(", " \( ", string)
#     string = re.sub(r"\)", " \) ", string)
#     string = re.sub(r"\?", " ? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip() if TREC else string.strip().lower()


# def clean_str_sst(string):
#     """ Tokenization/string cleaning for the SST dataset """
#     string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     return string.strip().lower()

#
# def build_toy_dataset():
#     """ 从真实数据集中生成toy数据集. 目前是顺序取n_train_ep个train实体对, n_test_ep个test实体对. """
#     print "Loading real dataset"
#     x = cPickle.load(open(conf.get('file_dir_path','processed_dataset'), "rb"))
#     real_dataset, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
#     print ' - dtype: ',W.dtype, W2.dtype
#     print 'Building toy dataset'
#     n_train_ep = 300
#     n_test_ep = 60
#
#     n_train_count = 0
#     n_test_count = 0
#     toy_dataset = list()
#     try:
#         for ep in real_dataset:
#             # if ep == {} :    # 这个放到后面处理, 因为在这里处理掉的话可能导致后面testset(n=1950)不是batch_size的整数倍.
#             #     print '[Warn] EP empty'
#             #     continue
#             if ep['ds_split'] == 'Train' and n_train_count < n_train_ep:
#                 toy_dataset.append(ep)
#                 n_train_count += 1
#             if ep['ds_split'] == 'Test' and n_test_count < n_test_ep:
#                 toy_dataset.append(ep)
#                 n_test_count += 1
#     except Exception, msg:
#         print ep
#         print msg
#     print 'Dumping...  ',
#     cPickle.dump([toy_dataset, W, W2, word_idx_map, vocab], open(conf.get('file_dir_path','processed_dataset_toy'), "wb"))
#     print 'finished'
#

# # 得到句子中的每个单词对应的index构成的list, list两边补充pad(=0)到最大长度.
# def idx_pad_sentence(sentence, word2idx_dict, max_l, pad = 0):
#     """ Transforms sentence into a list of indices. Pad with zeroes. """
#     x = []
#     n_pad = conf.getint("settings", "max_filter_h") - 1
#     for i in xrange(n_pad):
#         x.append(pad)
#     words = sentence.split()
#     for word in words:
#         x.append(word2idx_dict[word])        # 假设word就在word2idx_dict中.
#     while len(x) < max_l+2*n_pad:
#         x.append(pad)
#     return x


def idx_sentence(sentence, word2id_dict):
    """ Transforms sentence into a list of indices. Pad with zeroes. """
    x = []
    words = sentence.split()
    for word in words:
        x.append(word2id_dict[word])  # 假设word就在word2idx_dict中.
    return x


def pad_indexed_sentence(idx_sent, max_l, pad_mark):
    # max_filter_h = 5
    n_pad = conf.getint("settings", "max_filter_h") - 1

    n_after_pad = n_pad + max_l + n_pad

    p1 = [pad_mark] * n_pad
    p2 = [pad_mark] * (n_after_pad - n_pad - len(idx_sent))  # 后面还要补多少个
    return p1 + idx_sent + p2


def shape_datasets(built_eps, word2id_dict, max_l, n_relation_types):
    """ Transforms sentences into a 2-d matrix. """
    # min_pf = -n_pf/2
    # max_pf = n_pf/2    # 左闭右开
    # assert pf_pad < max_pf
    #
    # pf_pad_dict = dict()        # 为了复用idx_pad_sentence()
    # for i in range(min_pf, max_pf):
    #     pf_pad_dict[str(i)] = i

    train_x, train_y, test_x, test_y = [], [], [], []
    for ep in built_eps:  # 每个ep一个list. list中每个元素是ep对应的每个mention sentence对应的word list
        ep_sent_withPF = []
        for i in range(len(ep["sentences"])):
            sentence = ep["sentences"][i]
            indexed_sent = idx_sentence(sentence, word2id_dict)
            padded_idx_sent = pad_indexed_sentence(indexed_sent, max_l, 0)
            # pf1_str = " ".join([str(pfp[0]) for pfp in pf_sent])    # 为了复用idx_pad_sentence, 来给pf打pad.
            # pf2_str = " ".join([str(pfp[1]) for pfp in pf_sent])
            # padded_pf1 = pad_indexed_sentence(pf1_sent, max_l, pf_pad_mark)
            # padded_pf2 = pad_indexed_sentence(pf2_sent, max_l, pf_pad_mark)
            sent_withPF = [padded_idx_sent, ep["PFinfo"][i]]
            ep_sent_withPF.append(sent_withPF)
        # word_indexed_sents = [idx_pad_sentence(sentence, word2idx_dict, max_l) for sentence in ep["sentences"]]
        # labels转vec
        labels_vec = [0] * n_relation_types
        for label_idx in ep['labels']:  # 所以如果这时0s_as_NA=True, ep['labels']为空dict, label_vec全0.
            labels_vec[label_idx] = 1

        if ep["ds_split"] == "test":
            test_x.append(ep_sent_withPF)
            test_y.append(labels_vec)
        elif ep["ds_split"] == "train":
            train_x.append(ep_sent_withPF)
            train_y.append(labels_vec)
        else:
            logging.error('ds_split is {0}, not the required(train or test).'.format(ep["ds_split"]))
    return train_x, train_y, test_x, test_y


def read_some_negative_eps(filename, ds_split):
    # 得到negative_data的取值范围.
    if ds_split == 'train':
        neg_range = eval(conf.get('process_dataset', 'train_negative_range'))
    elif ds_split == 'test':
        neg_range = eval(conf.get('process_dataset', 'test_negative_range'))
    else:
        raise NotImplementedError
    if neg_range is None:
        return []  # extend一个空list得到的还是原来的list
    assert isinstance(neg_range, tuple) and len(neg_range) == 2

    print "Loading negative {0} data...   ".format(ds_split)
    eps_negative_all = read_from_file(filename, ds_split)
    if neg_range[0] == neg_range[1] == -1:
        eps_get = eps_negative_all
    else:
        eps_get = eps_negative_all[neg_range[0]: neg_range[1]]
    print 'PICK [{0}, {1}) eps from \'{2}\''.format(neg_range[0], neg_range[1], filename)
    return eps_get


# def generate_word2vec():
#     output_filename = r'D:/word2vec_input'
#     op_file = open(output_filename, 'wb')
#     corpus_dirname = r"D:/_Corpus/NYT00-07/"
#     years = ['2000','2001','2002','2003','2004','2005','2006','2007']
#     # years = ['2005','2006','2007']
#     # years = ['2005','2006']
#     for year in years:
#         year_dirname = corpus_dirname + year + '/'
#         tgz_files = filter(lambda filename: filename.endswith('.tgz'), os.listdir(year_dirname))
#         for tgz_file in tgz_files:
#             print 'Extracting {0}'.format(year+'-'+tgz_file)
#             tar = tarfile.open(year_dirname+tgz_file,'r')
#             xml_names = filter(lambda filename: filename.endswith('.xml'), tar.getnames())
#             for xml_name in xml_names:
#                 f = tar.extractfile(xml_name)
#                 # root = et.parse(f).getroot()
#                 raw_full_text = list()
#                 lines = f.readlines()
#                 for idx in range(len(lines)):
#                     # 只找正文. 但是发现很多文章只有标题.
#                     # if lines[idx].strip() == r'<block class="full_text">':
#                     #     idx+=1
#                     #     while idx<len(lines) and lines[idx].strip().startswith(r'<p>'):
#                     #         raw_full_text.append(lines[idx].strip().split(r'<p>')[1].split(r'</p>')[0])
#                     #         idx+=1
#                     #     break
#                     # 找所有<p>(分布在正文和标题)
#                     if lines[idx].strip().startswith(r'<p>'):
#                         raw_full_text.append(clean_str(lines[idx].strip().split(r'<p>')[1].split(r'</p>')[0])+' . ')
#                     if lines[idx].strip().startswith(r'<hl1>'):
#                         raw_full_text.append(clean_str(lines[idx].strip().split(r'<hl1>')[1].split(r'</hl1>')[0])+' . ')
#                 if len(raw_full_text) == 0:
#                     print 'Full_text or Title is not Found, at {0}{1}, content={2}'.format(year_dirname+tgz_file, xml_name, str(lines))
#                     print '-----------------------'
#                     continue
#                     # raise NotImplementedError, msg
#                 for line in raw_full_text:
#                     op_file.write(line)
#     op_file.close()


def parse_args():
    print 'Reading arguments:'
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='output dataset file path', type=str)
    # action='store_true'表示: 出现这个参数的时候存储的是True.
    parser.add_argument('--bin', action='store_true', default=False, help='convert to binary_rel: NA(0) and not NA(1).')
    parser.add_argument('--sl', action='store_true', default=False, help='single label (use the first) for trainset')
    args_dict = vars(parser.parse_args())

    if args_dict['o']:
        print '  - specify output .p file: {0}'.format(args_dict['o'])
    if args_dict['bin']:
        print '  - bin_rel = True'
    else:
        print '  - bin_rel = False'
    if args_dict['sl']:
        print '  - MISL(use first rel) for trainset, MIML for testset'
    else:
        print '  - MIML for trainset, MIML for testset'
    print '--------------------------'
    return args_dict


def process_data():
    args = parse_args()
    dataset_output_path = conf.get('file_dir_path', 'processed_dataset') if args['o'] is None else args['o']
    # bin_rel = args['b']
    # build_dataset(dataset_output_path, bin_rel)
    bin_rel = False
    print 'bin_rel is set \'False\' !'

    word2vec_filepath = conf.get('word_vector', 'w2v_file')
    train_positive_filepath = conf.get('file_dir_path', 'raw_train_positive_file')
    test_positive_filepath = conf.get('file_dir_path', 'raw_test_positive_file')
    train_negative_filepath = conf.get('file_dir_path', 'raw_train_negative_file')
    test_negative_filepath = conf.get('file_dir_path', 'raw_test_negative_file')

    # 1. 读入原始数据
    print "Loading positive train and test data... "
    # 把训练集和测试集都装进来, 以ep.ds_split="train"/"test"为识别标志.
    train_pos = read_from_file(train_positive_filepath, 'train')
    test_pos = read_from_file(test_positive_filepath, 'test')
    train_neg = read_some_negative_eps(train_negative_filepath, 'train')
    test_neg = read_some_negative_eps(test_negative_filepath, 'test')
    raw_eps = train_pos + test_pos + train_neg + test_neg
    dataset_status = {"rel2id_dict": MIMLEntityPair.REL2ID_DICT}
    print "Raw data loaded! n_eps = {0}".format(len(raw_eps))

    # 2. 从原始数据中构造要用的数据, 记为revs.
    print "Building from raw data ..."  #
    built_eps, vocab = build_data(raw_eps)  # vocab是单词及其出现次数的dict
    print "Data builted!"

    # 可选分支: 生成word2vec输入文件.
    # print "Branch -> Collecting gen_word2vec_input ..."
    # sents_all = []
    # tokens_all = set()
    # for ep in built_eps:
    #     if ep["ds_split"] != "train":
    #         continue
    #     for s in ep["sentences"]:
    #         sents_all.append(s.split())
    #         tokens_all.update(s.split())
    # print '{0} trainset sentences (which has {1} tokens) is being used as w2v input '.format(len(sents_all), len(tokens_all))
    # w2v_model = Word2Vec(sents_all, sg=1, size=50, min_count=1)
    # print "w2v_model built, len(vocab) = {0}, layer_1_size = {1}".format(len(w2v_model.vocab), w2v_model.layer1_size)
    # print "gensim word2vec model saved in: {0}".format(conf.get('word_vector', 'w2v_file'))
    # w2v_model.save(conf.get('word_vector', 'w2v_file'))
    # print "OK"
    # exit()

    # 3. 统计数据集信息.
    sent_lengths = []
    for built_ep in built_eps:
        for sent in built_ep["sentences"]:
            sent_lengths.append(len(sent.split()))
    max_sentence_l = np.array(sent_lengths).max()
    # max_sentence_l = 2000     # for test
    dataset_status['max_sentence_length'] = max_sentence_l
    print "Statistics:"
    print "  - number of entity pairs: " + str(len(built_eps))
    print "  - vocab size: " + str(len(vocab))
    print "  - max sentence length: " + str(max_sentence_l)
    # print ' -> conf: Set [dataset_stats]->max_sent_len to {0}'.format(max_sent_len)

    # 4. 生成"Word->词向量"的列表.
    n_word_dim = -1
    word_init_type = conf.get('word_vector', 'initialization')
    word2vec_dict = dict()  # 如果是word2vec模式, if段过后w2v_dict会被替换为填充了word2vec向量的词典.
    if word_init_type == "word2vec":
        print "Using: word2vec vectors"
        print "  - loading word2vec vectors...  ",
        word2vec_dict = load_word2vec_from_gsmodel(conf.get('word_vector', 'w2v_file'), vocab)
        n_word_dim = len(word2vec_dict.items()[0][1])
        print "  - n_word_dim: {0}, n_words already in word2vec: {1}".format(n_word_dim, len(word2vec_dict))
        assert n_word_dim == conf.getint('hpyer_parameters', 'wordv_length')
    elif word_init_type == "rand":
        n_word_dim = conf.getint('hpyer_parameters', 'wordv_length')
    elif word_init_type != 'word2vec' and word_init_type != 'rand':
        raise NotImplementedError, 'word embedding initialization mode not defined.'
    elif n_word_dim == -1:
        raise NotImplementedError, 'word_length not initializaed'
    else:
        raise NotImplementedError, 'wrong word_initialization method'
    # word2vec和rand模式的共有部分: 将vocab上'不在word2vec_dict上, 且最少出现min次'的单词的vec初始化为随机值.

    # print word2vec_dict.items()[0]
    # print word2vec_dict.items()[5]
    # print word2vec_dict.items()[12]
    # for k ,v in word2vec_dict.items():
    #     if k.startswith('$$'):
    #         print "{0}: vec={1}".format(k, v)

    add_unknown_words(word2vec_dict, vocab, n_word_dim)
    Wordv, word2id_dict = build_Wordv(word2vec_dict, n_word_dim)  # word2idx_dict从1开始.
    Wordv = np.asarray(Wordv, dtype=theano.config.floatX)
    print "Wordv shape= {0}".format(Wordv.shape)

    # 5. 将revs中的特征index化, 得到datasets.
    print 'Converting text dataset into index-based ...'
    if not bin_rel:
        datasets = shape_datasets(built_eps, word2id_dict, max_sentence_l, len(MIMLEntityPair.REL2ID_DICT))
    else:
        raise NotImplementedError
    # print '  - [! BINARY RELATION ] Converting to binary relations -> NA + notNA'
    # datasets = shape_datasets_bin(revs, word2idx_dict, max_sentence_l, len(MIMLEntityPair.REL_IDX_DICT))
    #
    # for i in range(len(datasets[3])):
    #     if np.sum(datasets[3][i])>1:
    #         print datasets[3][i]
    #
    # print '---------------'
    #
    # for i in range(len(datasets[1])):
    #     if np.sum(datasets[1][i])>1:
    #         print datasets[1][i]

    # (可选) 输出一份Zeng15输入格式的数据. 模块假设merge_ep = False. (在模块中有merge功能)
    print "Branch -> Converting to zeng\'15 input format"
    assert conf.getboolean('process_dataset', 'merge_ep') == False
    convert_mydata_2_zeng_format(datasets, Wordv, word2id_dict, dataset_status, l_limit=80, mention_dedup=True)
    exit()

    # 6. 输出到.p文件.
    print 'Outputing datasets to file: {0}'.format(dataset_output_path)
    cPickle.dump([datasets, Wordv, word2id_dict, vocab, dataset_status], open(dataset_output_path, "wb"))
    print "Finished!"


def convert_mydata_2_zeng_format(datasets, Wordv, word2id, dataset_status,
                                 op_dir="D:/Programming/simu_zeng15_data_withAll",
                                 l_limit=999999, mention_dedup=False):
    """ 注意: l_limit截取后, Wordv, word2id, vocab都不变. 只是其中有部分(id)不出现在数据集中而已."""

    rel2id = dataset_status["rel2id_dict"]
    max_l = dataset_status["max_sentence_length"]
    print 'max_sentence_length = ' + str(max_l)
    id2word = {}
    for k, v in word2id.items():
        id2word[v] = k
    id2rel = {}
    for k, v in rel2id.items():
        id2rel[v] = k

    print 'saving dict.txt'
    with open(op_dir + "/dict.txt", 'w') as op_dict_txt:
        for i in range(1, len(id2word) + 1):
            op_dict_txt.write(id2word[i] + '\n')

    print 'saving Wv.txt'
    with open(op_dir + "/Wv.txt", 'w') as op_Wv_txt:
        print 'Wordv[0] = ' + str(Wordv[0])
        for i in range(1, len(id2word) + 1):
            Wv_line = ' '.join(map(str, Wordv[i]))  # 很奇怪. str()之后Wordv[i]的精度会变.....
            op_Wv_txt.write(Wv_line + '\n')

    print 'saving rel2idx.txt'
    with open(op_dir + "/rel2id.txt", 'w') as op_rel2idx:
        for i in range(len(id2rel)):
            op_rel2idx.write(str(i) + ' ' + id2rel[i] + '\n')

    def convert_dataset_2Zeng(dataset_x, dataset_y, op_filename, l_limit, mention_dedup):
        if mention_dedup:
            print 'mention_deduplication = True'
        if l_limit != 999999:
            print 'l_limit = ' + str(l_limit)

        dataset_ordict = collections.OrderedDict()  # key是ep_line, value是[rel_line, m0, m1, m2,...] 其中rel_line是list()
        for i, ep in enumerate(dataset_x):
            tmp = np.nonzero(dataset_y[i])
            assert len(tmp) == 1 and len(tmp[0]) == 1
            y = tmp[0][0]

            n_mentions = len(ep)
            e1_got = None
            e2_got = None
            mentions_ready = []
            for mi, (mention, pf_info) in enumerate(ep):
                m_len, m_e1i, m_e2i = pf_info
                assert mention[4 + m_len - 1] != 0 and mention[4 + m_len] == 0  # 测试mention最后一个位置的idx是不是pf_info[0]-1
                mention_unpad = mention[4: 4 + m_len]
                if mi == 0:
                    e1_got = mention_unpad[m_e1i]
                    e2_got = mention_unpad[m_e2i]
                else:
                    assert e1_got == mention_unpad[m_e1i] and e2_got == mention_unpad[
                        m_e2i]  # 应该所有mention的e1, e2位置得到的结果都是一样的.
                if len(mention_unpad) <= l_limit:
                    mentions_ready.append(tuple([m_e1i, m_e2i] + mention_unpad))  # list是unhashable的type, 所以先转化为tuple.
            if mention_dedup and len(mentions_ready) > 0:  # 按顺序去重. (记得将n_mention重置)
                mentions_new = collections.OrderedDict()
                for m in mentions_ready:
                    mentions_new[m] = 1
                mentions_ready = mentions_new.keys()
                n_mentions = len(mentions_ready)
            assert e1_got is not None and e2_got is not None
            epline_str = "{0} {1}".format(e1_got, e2_got)
            if epline_str in dataset_ordict:
                # try:
                assert set(dataset_ordict[epline_str][1:]) == set(mentions_ready)
                # except AssertionError:
                #     print 'e1, e2= List({0})\t@\tList({1})'.format(id2word[e1_got], id2word[e2_got])
                #     print mentions_ready
                #     print dataset_ordict[epline_str]
                #     print dataset_ordict[epline_str][1:]
                #     exit()
                neg1i = dataset_ordict[epline_str][0].index(-1)
                dataset_ordict[epline_str][0][neg1i] = y
            else:
                rel_line = [y, -1, -1, -1, n_mentions]
                if len(mentions_ready) > 0:  # ==0表示通过l_limit过滤到这个ep没有合适的mention了. 这种情况直接不录入.
                    dataset_ordict[epline_str] = [rel_line] + mentions_ready

        with open(op_dir + "/" + op_filename, 'w') as op_dataset:
            for epline_str, relm_lines in dataset_ordict.items():
                op_dataset.write(epline_str + '\n')
                for line in relm_lines:
                    op_dataset.write(" ".join(map(str, line)) + '\n')

    # 单纯格式转换. 不去重(zeng15代码去重了).
    train_x, train_y, test_x, test_y = datasets
    print 'Converting datasets format to Zeng15Source ...'
    convert_dataset_2Zeng(train_x, train_y, 'jxt_train_convert.data', l_limit, mention_dedup)
    np_test_x, np_test_y = adjust_neg_pos_order(test_x, test_y,
                                                neg_before_pos=True)  # zeng15格式: test时pos在后, neg在前; train时pos在前, neg在后.
    convert_dataset_2Zeng(np_test_x, np_test_y, 'jxt_test_convert.data', l_limit, mention_dedup)


def adjust_neg_pos_order(dataset_x, dataset_y, neg_before_pos):
    assert conf.getboolean('process_dataset', '0s_as_NA') == False
    pos_x = []
    neg_x = []
    pos_y = []
    neg_y = []
    x_ready = []
    y_ready = []
    for ep, y in zip(dataset_x, dataset_y):
        if y[0] == 1:
            neg_x.append(ep)
            neg_y.append(y)
        else:
            pos_x.append(ep)
            pos_y.append(y)
    if neg_before_pos:
        x_ready.extend(neg_x)
        y_ready.extend(neg_y)
        x_ready.extend(pos_x)
        y_ready.extend(pos_y)
    else:
        x_ready.extend(pos_x)
        y_ready.extend(pos_y)
        x_ready.extend(neg_x)
        y_ready.extend(neg_y)
    return x_ready, y_ready


def process_data_from_zeng_input():
    """ 将zeng的txt数据处理成我的格式. """
    args = parse_args()

    # dataset_output_path = conf.get('file_dir_path', 'processed_dataset') if args['o'] is None else args['o']

    dataset_output_path = 'test.p' if args['o'] is None else args['o']

    trainset_sl = args['sl']
    print 'Processing Zeng15 .data to my format ...'
    wv_dim = 50
    max_l = 80

    # n_relations = 27
    n_relation_types = conf.getint("process_dataset", "n_relations")
    # zeng_rawdata_dir = data / Zeng_raw_data / gap_40_len_80
    input_dir = conf.get('file_dir_path', 'zeng_rawdata_dir')

    # train_data_path = input_dir + '/train_filtered.data'
    train_data_path = input_dir + '/train_filtered.data'

    test_data_path = input_dir + '/test_filtered.data'

    # 以数字隔开的数字
    Wordv_path = input_dir + '/50/wv.txt'
    # 词表大小
    dict_path = input_dir + '/50/dict.txt'

    # Read Wordv
    print 'Reading Wordv ...'
    with open(Wordv_path, 'r') as Wordv_file:
        allLines = Wordv_file.readlines()
        # 0行不用，最后一行初始化为0.5
        Wordv = np.zeros((len(allLines) + 2, wv_dim))
        i = 1
        for line in allLines:
            a = line.split(' ')
            print len(a)
            Wordv[i, :] = map(float, a)
            i += 1
        Wordv[i, :] = rng.uniform(low=-0.5, high=0.5, size=(1, wv_dim))

    Wordv = Wordv.astype(theano.config.floatX)

    # Read dataset
    def read_dataset(dataset_path, ds_split):
        data = []
        with open(dataset_path, 'r') as ds_file:
            while 1:
                PFinfo = []
                sentences_ready = []
                epline = ds_file.readline()
                if not epline:
                    break
                entities = map(int, epline.split(' '))
                relline = ds_file.readline()
                bagLabel = relline.split(' ')
                tmp = map(int, bagLabel[0:-1])
                rels = filter(lambda x: x >= 0, tmp)  # 已经过滤了占位符(-1).
                # 相关的句子数为n_mentions
                n_mentions = int(bagLabel[-1])  # 转化为int会把最后那个\n消掉.
                for i in range(n_mentions):
                    m_split = map(int, ds_file.readline().split(' '))
                    e1i = int(m_split[0])
                    e2i = int(m_split[1])
                    # 貌似所有句子都已3结尾
                    mention = m_split[2:-1]  # 先按照zeng的方法来吧, 不管最后一个.

                    PFinfo.append([len(mention), e1i, e2i])
                    sentences_ready.append(mention)

                rels_ready = [rels[0]] if ds_split == "train" and trainset_sl else rels  # 模仿zeng, 只取第一个label.
                data.append({"labels": rels_ready,  # int   只有一个rel
                             "ds_split": ds_split,  # 'train' / 'test'
                             "e1id": entities[0],
                             "e2id": entities[1],
                             "sentences": [s for s in sentences_ready],  # 句子的list.
                             "PFinfo": PFinfo,  # (sentence_length, e1_index, e2_idx)
                             })
        return data

    print 'Reading datasets ...'
    revs_zeng = read_dataset(train_data_path, 'train') + read_dataset(test_data_path, 'test')

    # 转化输入格式.
    print 'Converting format ...'
    train_x, train_y, test_x, test_y = [], [], [], []
    for ep in revs_zeng:
        ep_sent_withPF = []
        for i in range(len(ep["sentences"])):
            # 每个关系对有很对句子，取其中一句
            indexed_sent = ep["sentences"][i]
            # max_l = 80
            padded_idx_sent = pad_indexed_sentence(indexed_sent, max_l, 0)
            # padding完之后，每个句子88维，前4维，中间句子中每个词一维，最后全为0

            # 把每句话的句子信息和padding后的句子加入到ep_sent_withPF
            ep_sent_withPF.append([padded_idx_sent, ep["PFinfo"][i]])

        # n_relation_types = 27

        labels_vec = [0] * n_relation_types

        for label_idx in ep['labels']:  # 所以如果这时_NA_AS_0s=True, ep['labels']为空dict, label_vec全0.
            labels_vec[label_idx] = 1

        if ep["ds_split"] == "test":
            test_x.append(ep_sent_withPF)
            test_y.append(labels_vec)
        elif ep["ds_split"] == "train":
            train_x.append(ep_sent_withPF)
            train_y.append(labels_vec)
        else:
            logging.error('ds_split is {0}, not the required(train or test).'.format(ep["ds_split"]))

    print "printing statistics: "

    # 传进来的是label
    def n_pos_triples(dataset_y, check_sl=False):
        cnt = 0
        cnt_pos = 0
        for ep_y in dataset_y:
            ep_cnt = np.sum(ep_y)
            if check_sl:
                assert ep_cnt == 1
            cnt += ep_cnt
            if ep_y[0] != 1:
                # 如果关系开头不是1的话就加上这次关系的个数
                cnt_pos += ep_cnt
        return cnt, cnt_pos

    n_trainset_triples, n_trainset_pos_triples = n_pos_triples(train_y,
                                                               check_sl=True) if trainset_sl else n_pos_triples(train_y)
    n_testset_triples, n_testset_pos_triples = n_pos_triples(test_y)

    print '  - n_trainset_triples = {0}, pos = {1}'.format(n_trainset_triples, n_trainset_pos_triples)
    print '  - n_testset_triples = {0}, pos = {1}'.format(n_testset_triples, n_testset_pos_triples)

    print 'Outputing datasets to file: {0}'.format(dataset_output_path)
    print '  - datasets ...'
    datasets = [train_x, train_y, test_x, test_y]

    word2id = dict()
    with open(dict_path, 'r') as word_file:
        # 对可迭代函数 'iterable' 中的每一个元素应用‘function’方法，将结果作为list返回。
        word_list = map(lambda x: x.strip(), word_file.readlines())
        for i, word in enumerate(word_list):
            word2id[word] = i

    vocab = None

    dataset_status = {
        'max_sentence_length': max_l,
        'rel2id_dict': None,
    }

    cPickle.dump([datasets, Wordv, word2id, vocab, dataset_status], open(dataset_output_path, "wb"))

    # 先把每个训练数据取一个0-2随机数
    # 把整个测试数据分成3份
    # 第一份把取到0的数作为测试数据，取到1和2的数当做训练数据
    # 第二份把取到1的数作为测试数据，取到0和2的数当做训练数据
    # 第三份把取到2的数作为测试数据，取到0和1的数当做训练数据

    # n_cv = 3
    n_cv = conf.getint("process_dataset", "cv_folds")

    np.random.seed(3435)
    # 0-2之间的数一堆
    cv_alloc = np.random.randint(0, n_cv, len(train_x))

    for cv in range(n_cv):
        cv_train_x = []
        cv_train_y = []
        cv_test_x = []
        cv_test_y = []
        for i in xrange(len(cv_alloc)):
            if cv_alloc[i] == cv:
                cv_test_x.append(train_x[i])
                cv_test_y.append(train_y[i])
            else:
                cv_train_x.append(train_x[i])
                cv_train_y.append(train_y[i])
        print "  - cv_{0}: train_eps: {1}, test_eps: {2}".format(cv, len(cv_train_x), len(cv_test_x))
        cPickle.dump([[cv_train_x, cv_train_y, cv_test_x, cv_test_y],
                      Wordv, word2id, vocab, dataset_status], open(dataset_output_path + ".cv{0}".format(cv), "wb"))
    print "Finished!"


if __name__ == "__main__":
    # generate_word2vec()

    process_data_from_zeng_input()

    # process_data()

    # test_PF_function()
