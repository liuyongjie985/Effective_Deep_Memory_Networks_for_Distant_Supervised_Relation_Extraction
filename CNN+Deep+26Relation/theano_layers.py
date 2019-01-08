# -*- coding: utf-8 -*-

"""
Sample code for
Convolutional Neural Networks for Sentence Classification
http://arxiv.org/pdf/1408.5882v2.pdf

Much of the code is modified from
- deeplearning.net (for ConvNet classes)
- https://github.com/mdenil/dropout (for dropout)
- https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
"""

import numpy
import theano.tensor.shared_randomstreams
from theano.sandbox.rng_mrg import MRG_RandomStreams
import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d
from theano.tensor.nnet import conv
from theano.ifelse import ifelse
import ConfigParser
from lasagne.updates import norm_constraint

conf = ConfigParser.ConfigParser()
conf.read("ModelControl.ini")
theano.config.floatX = conf.get('theano_setup', 'floatX')


def _dropout(srng, input_data, input_shape, p):
    """  http://blog.csdn.net/stdcoutzyx/article/details/49022443  """
    mask = srng.binomial(n=1, p=1 - p, size=input_shape)
    output = input_data * T.cast(mask, theano.config.floatX)
    return output, mask


def my_dropout(srng, input_data, input_shape, p):
    """  http://blog.csdn.net/stdcoutzyx/article/details/49022443  """
    mask = srng.binomial(n=1, p=1 - p, size=input_shape)
    output = input_data * T.cast(mask, theano.config.floatX)
    return output, mask


def _get_srng_from_rng(rng):
    return T.shared_randomstreams.RandomStreams(rng.randint(999999))


class MLPDropout(object):
    # _MLP_SHAPE = [690,26] _MLP_ACTIVATIONS = [Iden] dropout_rates = [0.5]
    # layer_sizes = _MLP_SHAPE, activations = _MLP_ACTIVATIONS, dropout_rates = _DROPOUT_RATES
    def __init__(self, rng, layer_sizes, dropout_rates, activations):
        self.input = None
        # zip以最短序列作为长度
        # 690 × 26
        self.weight_matrix_sizes = zip(layer_sizes, layer_sizes[1:])
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        self.srng = _get_srng_from_rng(rng)
        # 列表
        self.dropout_rates = dropout_rates
        self.lrmask = None

        # 690 26
        n_in, n_out = self.weight_matrix_sizes[-1]
        dropout_output_layer = LogisticRegression(n_in=n_in, n_out=n_out)
        self.dropout_layers.append(dropout_output_layer)
        output_layer = LogisticRegression(n_in, n_out, W=dropout_output_layer.W * (1 - dropout_rates[-1]),
                                          b=dropout_output_layer.b)
        self.layers.append(output_layer)

        self.params = [param for layer in self.dropout_layers for param in layer.params]

    def feed(self, new_input, input_shape):
        self.input = new_input
        lr_input = new_input
        dropout_lr_input, self.lrmask = _dropout(self.srng, lr_input, input_shape,
                                                 self.dropout_rates[0])  # dropout the input

        # 正常的input用dropout了的层去算  ----原
        # 正常输入的就应该用正常的层算
        self.layers[-1].feed(lr_input)
        # dropout了的input用正常层去算-----原
        # dropout了的input用dropout了的层去算
        self.dropout_layers[-1].feed(dropout_lr_input)


class LogisticRegression(object):
    """
    Multi-class Logistic Regression Class.
    The logistic regression is fully described by a weight matrix :math:`W` and bias vector :math:`b`. Classification is done by
    projecting data points onto a set of hyperplanes, the distance to which is used to determine a class membership probability.
    """

    def __init__(self, n_in, n_out, W=None, b=None):
        """
        Initialize the parameters of the logistic regression
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """
        self.input = None
        self.score = None
        self.output = None

        self.W = theano.shared(value=numpy.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='LR_W') if W is None else W
        self.b = theano.shared(value=numpy.zeros((n_out,), dtype=theano.config.floatX), name='LR_b') if b is None else b

        self.params = [self.W, self.b]  # parameters of the model

    def feed(self, new_input):
        self.input = new_input
        self.score = T.dot(self.input, self.W) + self.b

        self.output = T.nnet.softmax(self.score)


class LeNetConvPoolLayer(object):
    def __init__(self, rng, filter_shape, poolsize, image_shape, non_linear):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps, filter height,filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps, image height, image width)
        """
        assert image_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        self.image_shape = image_shape

        self.input = None
        self.conv_out = None
        self.nonlinear_out = None
        self.output = None

        if self.non_linear == "none" or self.non_linear == "relu":
            W_bound = 0.01
        else:
            # 1×3×60
            fan_in = numpy.prod(filter_shape[1:])
            # 230×3×60 / 86×1
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
            # /0.00907172996  0.0952456296
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        # 230, 1, 3, 60
        # 正负0.0952456296的平均分布
        W_values = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, borrow=True, name="CP_W")

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        # conv_bias = True
        if conf.getboolean("hpyer_parameters", "conv_bias"):
            self.b = theano.shared(value=b_values, borrow=True, name="CP_b")
            self.params = [self.W, self.b]
        else:
            self.b = T.constant(b_values, name="CP_b")
            self.params = [self.W]

    def feed(self, new_input):
        """
        predict for new data.

        :type new_input: theano.tensor.dtensor4
        :param new_input: symbolic image tensor, of shape image_shape
        """
        # conv
        self.input = new_input
        self.conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape,
                                    image_shape=self.image_shape)
        if self.non_linear.lower() == "tanh":
            # b = 1×230×1×1
            self.nonlinear_out = T.tanh(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif self.non_linear.lower() == "relu":
            self.nonlinear_out = T.nnet.relu(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            raise NotImplementedError

        # pooling 86×1
        pooling_out = pool_2d(input=self.nonlinear_out, ds=self.poolsize, ignore_border=False, mode='max')

        self.output = pooling_out

    # Piecewise pooling
    def piecewisePooling_feed(self, new_input):
        # mentions_batch = 句子数×1×88×60
        # eli_batch = 句子数×1
        mentions_batch, e1i_batch, e2i_batch = new_input
        # conv
        # input = 句子数×1×88×60
        # filter = 230×1×3×60
        self.conv_out = conv.conv2d(input=mentions_batch, filters=self.W, filter_shape=self.filter_shape,
                                    image_shape=self.image_shape)

        # conv_out=句子数×230×86×1

        # nonlinear_out = 句子数×230×86×1
        if self.non_linear.lower() == "tanh":
            # b是0
            self.nonlinear_out = T.tanh(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif self.non_linear.lower() == "relu":
            self.nonlinear_out = T.nnet.relu(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            raise NotImplementedError

        # pooling
        # filter_h = 3
        filter_h = self.filter_shape[2]
        # n_pad_head = 4
        n_pad_head = conf.getint('settings', 'max_filter_h') - 1
        assert n_pad_head == 4
        # numpy.floor向上取整
        # 反正偏移了3个单位
        idx_shift = n_pad_head - int(numpy.floor(filter_h / 2))  # 经过pad和convolution后, 相对于e1i, e2i(pad前)偏移了多少.

        e1i_conved = e1i_batch + idx_shift
        e2i_conved = e2i_batch + idx_shift

        # 得到每个句子左实体的位置 + 1和右实体的位置 + 1
        [m_seg2_st_batch, m_seg3_st_batch], _ = \
            theano.scan(fn=lambda e1i, e2i: ifelse(T.lt(e1i, e2i), (e1i + 1, e2i + 1), (e2i + 1, e1i + 1)),
                        sequences=[e1i_conved, e2i_conved])

        nonlinear_out_3d = self.nonlinear_out.flatten(3)

        def piecewise_pooling(conved_m, m_seg2_st, m_seg3_st):
            seg1_out = T.max(conved_m[:, :m_seg2_st], axis=1)
            seg2_out = T.max(conved_m[:, m_seg2_st: m_seg3_st], axis=1)
            seg3_out = T.max(conved_m[:, m_seg3_st:], axis=1)
            return T.transpose(T.stack((seg1_out, seg2_out, seg3_out))).flatten()

        # 对于每一个句子返回一个230×3的向量
        pooling_2d, _ = theano.scan(fn=piecewise_pooling,
                                    sequences=[nonlinear_out_3d, m_seg2_st_batch, m_seg3_st_batch])

        self.input = new_input
        self.output = pooling_2d


class TwoConvPoolLayers(object):
    def __init__(self, rng, cp1_filter_shape, cp1_pool_size, cp2_filter_shape, cp2_pool_size, image_shape,
                 cp1_nonlinear, cp2_nonlinear):
        self.cp1 = LeNetConvPoolLayer(rng, cp1_filter_shape, cp1_pool_size, image_shape, cp1_nonlinear)
        cp1_fm_img_h = image_shape[2] - cp1_filter_shape[2] + 1
        cp2_img_h = int(numpy.ceil(cp1_fm_img_h / float(cp1_pool_size[0])))
        cp2_image_shape = [None, cp1_filter_shape[0], cp2_img_h, 1]
        self.cp2 = LeNetConvPoolLayer(rng, cp2_filter_shape, cp2_pool_size, cp2_image_shape, cp2_nonlinear)

        self.input = None
        self.output = None
        self.params = self.cp1.params + self.cp2.params

    def feed(self, new_input):
        self.input = new_input
        self.cp1.feed(new_input)
        cp1_out = self.cp1.output
        self.cp2.feed(cp1_out)
        self.output = self.cp2.output


class MyLogisticRegression(object):
    """
    Multi-class Logistic Regression Class.
    The logistic regression is fully described by a weight matrix :math:`W` and bias vector :math:`b`. Classification is done by
    projecting data points onto a set of hyperplanes, the distance to which is used to determine a class membership probability.
    """

    # def __init__(self, n_in, n_out, W1=None, b1=None, W2=None, b2=None, W3=None, b3=None,
    #              W4=None, b4=None,
    #              W5=None, b5=None, W6=None, b6=None, W7=None, b7=None, W8=None, b8=None,
    #              W9=None, b9=None, W10=None, b10=None, W11=None, b11=None, W12=None, b12=None,
    #              W13=None, b13=None, W14=None, b14=None, W15=None, b15=None, W16=None, b16=None,
    #              W17=None, b17=None, W18=None, b18=None, W19=None, b19=None, W20=None, b20=None,
    #              W21=None, b21=None, W22=None, b22=None, W23=None, b23=None, W24=None, b24=None,
    #              W25=None, b25=None, W26=None, b26=None):

    def __init__(self, n_in, n_out, W=None, b=None):
        """
        Initialize the parameters of the logistic regression
        :type n_in: int
        :param n_in: number of input units, the dimension of the space in which the datapoints lie
        :type n_out: int
        :param n_out: number of output units, the dimension of the space in which the labels lie
        """
        self.input = None
        self.score = None
        self.output = None

        self.W = theano.shared(value=numpy.zeros((26, n_in, n_out), dtype=theano.config.floatX),
                               name='LR_W') if W is None else W
        self.b = theano.shared(value=numpy.zeros((26, n_out), dtype=theano.config.floatX),
                               name='LR_b') if b is None else b

        self.params = [self.W, self.b]

    def feed(self, idx, new_input):
        self.input = new_input
        # (50,2)
        self.score = T.dot(self.input, self.W[idx]) + self.b[idx]
        # (50,2)
        self.output = T.nnet.softmax(self.score)
        p = self.output[:, 0]
        return p

        # self.output = T.nnet.sigmoid(self.score)
        # return self.output


class MyMLPDropout(object):
    def __new__(cls, *args, **kw):
        if not hasattr(cls, '_instance'):
            orig = super(MyMLPDropout, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        return cls._instance

    # _MLP_SHAPE = [690,26] _MLP_ACTIVATIONS = [Iden] dropout_rates = [0.5]
    # layer_sizes = _MLP_SHAPE, activations = _MLP_ACTIVATIONS, dropout_rates = _DROPOUT_RATES

    def __init__(self, rng, layer_sizes, dropout_rates, activations):
        self.input = None
        # 690 × 2f
        self.weight_matrix_sizes = layer_sizes
        self.layers = []
        self.dropout_layers = []
        self.activations = activations
        self.srng = _get_srng_from_rng(rng)
        # 列表
        self.dropout_rates = dropout_rates
        self.lrmask = None

        # 690 26
        n_in, n_out = self.weight_matrix_sizes[-1]
        assert n_in == (690 + 120)
        assert n_out == 2
        self.dropout_layers.append(MyLogisticRegression(n_in=n_in, n_out=n_out))

        self.layers.append(MyLogisticRegression(n_in, n_out, W=self.dropout_layers[0].W * (1 - dropout_rates[-1]),
                                                b=self.dropout_layers[0].b))

        self.params = [param for layer in self.dropout_layers for param in layer.params]

    # new_input = (26, 50, (690+120))
    # input_shape = (26,50,(690+120))
    def feed(self, idx, new_input, input_shape):
        self.input = new_input
        lr_input = new_input

        # 这个dropout出了问题，原本是50乘690的，现在要变成26乘50乘690

        dropout_lr_input, self.lrmask = _dropout(self.srng, lr_input, input_shape,
                                                 self.dropout_rates[0])  # dropout the input

        # 那我现在不drop了
        # (50,(690+120))
        # self.lrmask = self.srng.binomial(n=1, p=1 - self.dropout_rates[0], size=input_shape)

        # 正常的input用dropout了的层去算  ----原
        # 正常输入的就应该用正常的层算
        def normalForEveryRelation(myi, lr_input_item):
            # (50)
            temp = self.layers[0].feed(myi, lr_input_item)
            # return temp[:,0]
            return temp

        normal_re, normal_up = theano.scan(normalForEveryRelation, sequences=[idx, lr_input])

        def dropoutForEveryRelation(myi, dropout_lr_input_item):
            temp = self.dropout_layers[0].feed(myi, dropout_lr_input_item)
            # return temp[:,0]
            return temp

        dropout_re, dropout_up = theano.scan(dropoutForEveryRelation, sequences=[idx, dropout_lr_input])

        return [normal_re, dropout_re]


class MyLeNetConvPoolLayer(object):
    def __init__(self, rng, filter_shape, poolsize, image_shape, non_linear):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps, filter height,filter width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps, image height, image width)
        """
        assert image_shape[1] == filter_shape[1]

        self.filter_shape = filter_shape
        self.poolsize = poolsize
        self.non_linear = non_linear
        self.image_shape = image_shape

        self.input = None
        self.conv_out = None
        self.nonlinear_out = None
        self.output = None

        if self.non_linear == "none" or self.non_linear == "relu":
            W_bound = 0.01
        else:
            # 1×3×60
            fan_in = numpy.prod(filter_shape[1:])
            # 230×3×60 / 86×1
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
            # /0.00907172996  0.0952456296
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))

        # 230, 1, 3, 60
        # 正负0.0952456296的平均分布
        W_values = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX)

        self.W = theano.shared(value=W_values, borrow=True, name="CP_W")

        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)

        # conv_bias = True
        if conf.getboolean("hpyer_parameters", "conv_bias"):
            self.b = theano.shared(value=b_values, borrow=True, name="CP_b")
            self.params = [self.W, self.b]
        else:
            self.b = T.constant(b_values, name="CP_b")
            self.params = [self.W]

    def feed(self, new_input):
        """
        predict for new data.

        :type new_input: theano.tensor.dtensor4
        :param new_input: symbolic image tensor, of shape image_shape
        """
        # conv
        self.input = new_input
        self.conv_out = conv.conv2d(input=self.input, filters=self.W, filter_shape=self.filter_shape,
                                    image_shape=self.image_shape)
        if self.non_linear.lower() == "tanh":
            # b = 1×230×1×1
            self.nonlinear_out = T.tanh(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif self.non_linear.lower() == "relu":
            self.nonlinear_out = T.nnet.relu(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            raise NotImplementedError

        # pooling 86×1
        pooling_out = pool_2d(input=self.nonlinear_out, ds=self.poolsize, ignore_border=False, mode='max')

        self.output = pooling_out

    # Piecewise pooling
    def piecewisePooling_feed(self, new_input):
        # mentions_batch = 句子数×1×88×60
        # eli_batch = 句子数×1
        mentions_batch, e1i_batch, e2i_batch = new_input

        # conv
        # input = 句子数×1×88×60
        # filter = 230×1×3×60
        self.conv_out = conv.conv2d(input=mentions_batch, filters=self.W, filter_shape=self.filter_shape,
                                    image_shape=self.image_shape)

        # conv_out=句子数×230×86×1 已验证

        # nonlinear_out = 句子数×230×86×1
        if self.non_linear.lower() == "tanh":
            # b是0
            self.nonlinear_out = T.tanh(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        elif self.non_linear.lower() == "relu":
            self.nonlinear_out = T.nnet.relu(self.conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        else:
            raise NotImplementedError

        # pooling
        # filter_h = 3
        filter_h = self.filter_shape[2]
        # n_pad_head = 4
        n_pad_head = conf.getint('settings', 'max_filter_h') - 1
        assert n_pad_head == 4
        # numpy.floor向上取整
        # 反正偏移了3个单位
        idx_shift = n_pad_head - int(numpy.floor(filter_h / 2))  # 经过pad和convolution后, 相对于e1i, e2i(pad前)偏移了多少.

        e1i_conved = e1i_batch + idx_shift
        e2i_conved = e2i_batch + idx_shift

        # 得到每个句子左实体的位置 + 1和右实体的位置 + 1
        [m_seg2_st_batch, m_seg3_st_batch], _ = \
            theano.scan(fn=lambda e1i, e2i: ifelse(T.lt(e1i, e2i), (e1i + 1, e2i + 1), (e2i + 1, e1i + 1)),
                        sequences=[e1i_conved, e2i_conved])

        nonlinear_out_3d = self.nonlinear_out.flatten(3)

        def piecewise_pooling(conved_m, m_seg2_st, m_seg3_st):
            seg1_out = T.max(conved_m[:, :m_seg2_st], axis=1)
            seg2_out = T.max(conved_m[:, m_seg2_st: m_seg3_st], axis=1)
            seg3_out = T.max(conved_m[:, m_seg3_st:], axis=1)
            return T.transpose(T.stack((seg1_out, seg2_out, seg3_out))).flatten()

        # 对于每一个句子返回一个230×3的向量
        pooling_2d, _ = theano.scan(fn=piecewise_pooling,
                                    sequences=[nonlinear_out_3d, m_seg2_st_batch, m_seg3_st_batch])

        self.input = new_input
        self.output = pooling_2d

        return mentions_batch
