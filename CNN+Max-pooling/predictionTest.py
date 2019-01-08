import theano
import theano.tensor as T
import numpy as np

rng = np.random.RandomState(3435)

d = T.dvector()

re, up = theano.scan(lambda p_ep: p_ep > 0.5, sequences=[d])

myf = theano.function([d], re)

ok = myf(np.asarray(rng.uniform(low=-1, high=1, size=[10]), dtype=theano.config.floatX))

print ok
