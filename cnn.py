# -*- coding: utf-8 -*-
import numpy as np
#from Levenshtein import distance
from tweet_sent_analysis import XML_parser
from gensim.models.word2vec import KeyedVectors
import theano
import theano.tensor as T
import lasagne
from sklearn.model_selection import train_test_split
import time

def max_len_sent(filename):
    '''
    Вычисление максимальной длины твита в обучающем множестве.
    '''
    parser = XML_parser()
    tweets, _ = parser.xml_parse(filename)
    tweets_tokenized = [tweet.split() for tweet in tweets]
    max_len = len(tweets_tokenized[0])
    for i in range(1,len(tweets_tokenized)):
        if len(tweets_tokenized[i]) > max_len:
            max_len = len(tweets_tokenized[i])
    # print(tweets_tokenized[0])
    # print(max_len)
    return max_len

def load_model(model_name):
    '''
    Загрузка модели в формате w2v.
    '''
    model = KeyedVectors.load_word2vec_format(model_name, binary=True, unicode_errors='ignore')
    model.init_sims(replace=True)
    print('The model is loaded.')
    return model

class DataProcessing():
    
    def __init__(self):
        self.parser = XML_parser()

    def load_data(self, filename):
        '''
        Функция извлекает список твитов и numpy-массив меток классов из выборки формата xml.
        '''
        return self.parser.xml_parse(filename)
        
    def sent_vectorize(self, model, tweets, max_len):
        '''
        '''
        X = np.zeros(shape=(len(tweets),1,max_len,100), dtype = np.float32)
        for i, tweet in enumerate(tweets):
            translator = str.maketrans('', '', ',.'":;!?_")
            tweet_w_punct = tweet.translate(translator)
            words = tweet_w_punct.split()
            tweet_array = np.zeros(shape=(max_len,100))
            for j, word in enumerate(words):
                try:
                    tweet_array[j] = model[word.lower()]
                except KeyError:
                    tweet_array[j] = 0
            X[i][0] = tweet_array
        return X

    def split_sets(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
        return X_train, X_test, y_train, y_test

class CNN():

    def build_cnn(self, max_len, input_var=None):
        network = lasagne.layers.InputLayer(shape=(None, 1, max_len, 100),
                                            input_var=input_var)
        # Convolutional layer with 32 kernels of size 5x5.
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        # Max-pooling layer of factor 2 in both dimensions:
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

        # A fully-connected layer of 256 units with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

        # And, finally, the 10-unit output layer with 50% dropout on its inputs:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

        return network

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)

        yield inputs[excerpt], targets[excerpt]

    def main(self, train_filename, test_filename, num_epochs=500):
        # Load the dataset
        print("Loading data...")
        model = load_model("all.norm-sz100-w10-cb0-it1-min100.w2v")

        max_len = max_len_sent(train_filename)

        vectorizer = DataProcessing()
        print("Train set processing...")
        X_for_train, y_for_train = vectorizer.load_data(train_filename) #('test_xml.xml')
        X_tweets_train, X_tweets_val, target_train, target_val = vectorizer.split_sets(X_for_train,y_for_train)
        y_train = np.array(target_train, dtype=np.int32) + 1
        y_val = np.array(target_val, dtype=np.int32) + 1
        X_train = vectorizer.sent_vectorize(model, X_tweets_train, max_len)
        X_val = vectorizer.sent_vectorize(model, X_tweets_val, max_len)
        print("Train and validation sets are created.\nTest set processing...")
        X_tweets_test, target_test = vectorizer.load_data(test_filename)
        X_test = vectorizer.sent_vectorize(model, X_tweets_test, max_len)
        y_test = np.array(target_test, dtype=np.int32) + 1
        print("Test set is created.")

        # Prepare Theano variables for inputs and targets
        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')

        # Create model
        print("Building model and compiling functions...")
        network = self.build_cnn(max_len, input_var)
        # Create a loss expression for training, i.e., a scalar objective we want
        # to minimize (for our multi-class problem, it is the cross-entropy loss):
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                                target_var)
        test_loss = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                          dtype=theano.config.floatX)

        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

        # Finally, launch the training loop.
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in self.iterate_minibatches(X_train, y_train, 500, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in self.iterate_minibatches(X_val, y_val, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

        # After training, we compute and print the test error:
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in self.iterate_minibatches(X_test, y_test, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results:")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))

if __name__ == '__main__':
    network = CNN()
    network.main('bank_train_2016.xml', 'banks_test_etalon.xml')