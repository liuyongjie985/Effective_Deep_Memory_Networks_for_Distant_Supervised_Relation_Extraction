import theano
import theano.tensor as T
from theano.tensor.nnet import conv



conv_out = conv.conv2d(input=mentions_batch, filters=self.W, filter_shape=self.filter_shape,
                                    image_shape=self.image_shape)