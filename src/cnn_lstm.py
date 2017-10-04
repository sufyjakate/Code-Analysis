import tensorflow as tf
import numpy
import tflearn
from keras.layers import Dense, Dropout, regularizers, Embedding, LSTM, Bidirectional, Conv1D, GlobalMaxPooling1D, \
    BatchNormalization, Flatten, GaussianNoise
from keras.models import Sequential
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras.utils import np_utils
from pip.req.req_file import preprocess

numpy.random.seed(7)

class Code_Completion_Baseline:


    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        #vector = np_utils.to_categorical((xs, ys), 1)
        return vector

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        for token_list in token_lists:
            for token in token_list:
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(previous_token_string))
                    ys.append(self.one_hot(token_string))
        (xs, ys) = sequence.pad_sequences((xs, ys), maxlen=6, padding='pre', value=0.33)
        print("x,y pairs: " + str(len(xs)))
        xs = xs.reshape(6, -1)
        ys = ys.reshape(6, -1)
        #(xtrain, ytrain) = numpy.array((xs, ys))
        #xtrain.reshape(-1, 7396)
        #ytrain.reshape(-1, 7396)
        return (xs, ys)

    def create_network(self):
        feature_size = 5000
        length = 400
        dims = 32
        filters = 250
        kernel_size = 3
        hidden = 250

        self.model = Sequential()

        self.model.add(Embedding(feature_size, dims, input_length=len(self.string_to_number)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv1D(filters, kernel_size, padding='causal', activation='tanh', kernel_initializer='he_uniform', kernel_regularizer=l2(0.00001)))
        self.model.add(BatchNormalization())

        self.model.add(GlobalMaxPooling1D())

        self.model.add(Dense(hidden, activation='sigmoid'))
        self.model.add(GaussianNoise(0.5))
        self.model.add(Dropout(0.25))

        self.model.add(Dense(hidden, activation='sigmoid'))
        self.model.add(Dropout(0.8))
        #self.model.add(Flatten())
        self.model.add(BatchNormalization())
        self.model.add(Dense(len(self.string_to_number), activation='softmax'))

        active = RMSprop()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=active, metrics=['accuracy'])
        print(self.model.summary())

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        #self.model.load(model_file)
        self.model.load_weights(model_file)
    def train(self, token_lists, model_file):
        batch_size = 256
        epochs = 500

        (xtrain, ytrain) = self.prepare_data(token_lists)
        #(xs, ys) = numpy.reshape((xs, ys),(-1, self.string_to_number))
        self.create_network()
        self.model.fit(xtrain, ytrain, epochs=epochs, batch_size=batch_size, validation_split=0.33)
        self.model.save(model_file)
        #score = self.model.evaluate(xs, ys)
        #print("\n%s: %.2f%%" % (self.model[1], score[1] * 100))

    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        x = numpy.array(x)
        x = x.reshape(6, -1)
        x = sequence.pad_sequences(x, maxlen=len(self.string_to_number), padding='pre', value=0.33)
        y = self.model.predict([x])
        #y = self.model.evaluate([x], verbose=0)
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
