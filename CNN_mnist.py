import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
import numpy as np
import gzip
import cPickle as pickle
import pylab
from PIL import Image
floatX = theano.config.floatX

class ConvPoolLayer(object):
    def __init__(self, input, filter_shape, img_shape, pool_size=(2, 2), prex='ConvPoolLayer'):
        self.input = input
        fan_in = np.prod(filter_shape[1:])
        fan_out = filter_shape[0] * np.prod(filter_shape[2:]) // np.prod(pool_size)
        w_bound = np.sqrt(6. / (fan_in + fan_out))

        self.W = theano.shared(value=np.asarray(
            np.random.uniform(
                low=-w_bound,
                high=w_bound,
                size=filter_shape
            ),
            dtype=floatX),
            name=prex + 'W',
            borrow=True
        )
        b_size = (filter_shape[0],)
        self.b = theano.shared(value=np.zeros(b_size, dtype=floatX),
                               name=prex + 'b',
                               borrow=True
                               )
        self.params = [self.W, self.b]
        conv_out = conv2d(
            input=input,
            filters=self.W,
            input_shape=img_shape,
            filter_shape=filter_shape
        )
        pool_out = pool.pool_2d(
            input=conv_out,
            ws=pool_size,
            ignore_border=True
        )
        self.output = T.tanh(pool_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, prex='HiddenLayer'):
        self.input = input
        self.W = theano.shared(
            value=np.random.uniform(
                low=-1/np.sqrt(n_in + n_out),
                high=1/np.sqrt(n_in + n_out),
                size=(n_in, n_out),
            ),
            name=prex + 'W',
        )
        self.b = theano.shared(
            value=np.zeros(n_out, dtype=floatX),
            name=prex + 'b',
        )
        self.params = [self.W, self.b]
        self.output = T.tanh(T.dot(input, self.W) + self.b)


class LogRegressionLayer(object):
    def __init__(self, input, n_in, n_out, prex='LogRegerssionLayer'):
        self.input = input
        self.W = theano.shared(
            value=np.random.uniform(
                low=-1/np.sqrt(n_in + n_out),
                high=1/np.sqrt(n_in + n_out),
                size=(n_in, n_out)
            ),
            name=prex + 'W',
        )
        self.b = theano.shared(
            value=np.zeros(n_out, dtype=floatX),
            name=prex + 'b',
        )
        self.params = [self.W, self.b]
        self.y_pred_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.y_pred_given_x, axis=1)

    def neg_log_likelihood(self, y):
        return -T.mean(T.log(self.y_pred_given_x)[T.arange(y.shape[0]), y])
    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

def load_data(datapath):
    with gzip.open(datapath, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
    def shared_dataset(dataxy):
        data_x, data_y = dataxy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=floatX),
                                 borrow=True)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=floatX),
                                 borrow=True)
        return shared_x, T.cast(shared_y, 'int32')
    train_x, train_y = shared_dataset(train_set)
    test_x, test_y = shared_dataset(test_set)
    valid_x, valid_y = shared_dataset(valid_set)
    rval = [(train_x, train_y), (test_x, test_y),
            (valid_x, valid_y)]
    return rval

class LeNet(object):
    def __init__(self, input, y, nkerns, n_batch):
        self.input = input
        layer0 = ConvPoolLayer(
            input=input,
            filter_shape=(nkerns[0], 1, 5, 5),
            img_shape=(n_batch, 1, 28, 28),
            pool_size=(2, 2)
        )
        layer1 = ConvPoolLayer(
            input=layer0.output,
            filter_shape=(nkerns[1], nkerns[0], 5, 5),
            img_shape=(n_batch, nkerns[0], 12, 12),
            pool_size=(2, 2)
        )
        layer2_input = layer1.output.flatten(2)
        layer2 = HiddenLayer(
            input=layer2_input,
            n_in=nkerns[1] * 4 * 4,
            n_out=500
        )
        layer3 = LogRegressionLayer(
            input=layer2.output,
            n_in=500,
            n_out=10
        )
        self.cost = layer3.neg_log_likelihood(y)
        self.errors = layer3.errors(y)
        self.params = layer3.params + layer2.params + layer1.params + layer0.params
        self.y_pred = layer3.y_pred

def LeNet_mnist(datapath='./data/mnist.pkl.gz',
                learning_rate=0.01,
                n_epochs=1000,
                batch_size=600):
    print 'Loading data ......'
    data = load_data(datapath)
    train_x, train_y = data[0]
    test_x, test_y = data[1]
    valid_x, valid_y = data[2]
    n_batches_train = train_x.get_value(borrow=True).shape[0] // batch_size
    n_batches_test = test_x.get_value(borrow=True).shape[0] // batch_size
    n_batches_valid = valid_x.get_value(borrow=True).shape[0] // batch_size
    print 'Building model ......'
    x = T.matrix('x')
    y = T.ivector('y')
    index = T.lscalar()
    input = x.reshape((batch_size, 1, 28, 28))
    classifier = LeNet(input=input,
                       y=y,
                       nkerns=(5, 50),
                       n_batch=batch_size)
    cost = classifier.cost

    errors = classifier.errors
    params = classifier.params

    grads = T.grad(cost, params)
    updates = [(param_i, param_i-learning_rate*grad_i)
              for param_i, grad_i in zip(params, grads)]

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={x: train_x[index * batch_size: (index + 1) * batch_size],
                y: train_y[index * batch_size: (index + 1) * batch_size]}
    )
    test_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens={x: test_x[index * batch_size: (index + 1) * batch_size],
                y: test_y[index * batch_size: (index + 1) * batch_size]}
    )
    valid_model = theano.function(
        inputs=[index],
        outputs=errors,
        givens={x: valid_x[index * batch_size: (index + 1) * batch_size],
                y: valid_y[index * batch_size: (index + 1) * batch_size]}
    )
    print 'Training ......'
    patience = 5000
    patience_increase = 2
    improve_threshold = 0.95
    valid_frequency = min(patience//2, n_batches_train)
    epoch = 0
    loop_flag = False
    best_valid_error = np.inf
    while(epoch<n_epochs) and (loop_flag==False):
        epoch += 1
        for minibatch in range(n_batches_train):
            cost = train_model(minibatch)
            #print cost
            iter = minibatch + (epoch - 1) * n_batches_train
            if (iter + 1) % valid_frequency == 0:
                valid_errors = [valid_model(i) for i in range(n_batches_valid)]
                this_valid_error = np.mean(valid_errors)
                print ('epoch %i, minibatch %i/%i, valid_error: %f%%' %
                       (epoch,
                        minibatch + 1,
                        n_batches_train,
                        this_valid_error * 100)
                       )
                if this_valid_error < best_valid_error:
                    if this_valid_error < best_valid_error * improve_threshold:
                        patience = max(patience, iter * patience_increase)
                    test_score = [test_model(i)
                                  for i in range(n_batches_test)]
                    best_test_score = np.mean(test_score)
                    print('epoch %i, test_score: %f%%' %
                          (epoch,
                           best_test_score * 100)
                          )
                    with open('best_lenet_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)
                    best_valid_error = this_valid_error

            if patience <= iter:
                loop_flag = True
                break
    print 'Optimization Done!'


def predict(img):
    classifier = pickle.load(open('best_lenet_model.pkl'))
    predict_model = theano.function(inputs=[classifier.input],
                                    outputs=classifier.y_pred)
    predict_values = predict_model(img)
    print predict_values
    pylab.subplot(1,3,1); pylab.imshow(img(0,0,28,28)); pylab.gray()
    pylab.subplot(1, 3, 2)
    pylab.imshow(img(1, 0, 28, 28))
    pylab.gray()
    pylab.subplot(1, 3, 3)
    pylab.imshow(img(2, 0, 28, 28))
    pylab.gray()


if __name__ == '__main__':
    LeNet_mnist()
    data = load_data('./data/mnist.pkl.gz')
    test_x, test_y = data[1]
    print test_x.get_value(borrow=True).shape
    img = test_x[2:5, :]
    img = img.reshape((3, 1, 28, 28))
    predict(img)