import tensorflow as tf
import numpy
import tflearn
from keras import Input
from keras.layers import Dense, Dropout, regularizers, Embedding, LSTM, Bidirectional, TimeDistributed, Flatten, \
    BatchNormalization, GlobalMaxPooling1D, GaussianNoise, AveragePooling1D
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, nadam
from keras.preprocessing import sequence

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
        (xs, ys) = sequence.pad_sequences((xs, ys), maxlen=86, padding='pre', value=0.0)
        print("x,y pairs: " + str(len(xs)))
        xs = xs.reshape(xs.shape[0], xs.shape[1])
        #xs = xs.reshape(6, -1)
        #ys = ys.reshape(6, -1)

        #(xs, ys) = (xs, ys).reshape([-1, 28, 28, 1])
        #(xs, ys) = Input(shape=(xs, ys).shape, name=(xs, ys))
        return (xs, ys)

    def create_network(self):
        feature_size = 5000
        length = 400
        dims = 32
        filters = 250
        kernel_size = 3
        hidden = 86

        self.model = Sequential()

        self.model.add(Embedding(feature_size, dims, input_length=86))
        #self.model.add(Dropout(0.5))

        self.model.add(LSTM(100))
        #self.model.add(Flatten())
        #self.model.add(TimeDistributed(Dense(2)))
        #self.model.add(AveragePooling1D())

        #self.model.add(Flatten())
        self.model.add(Dense(len(self.string_to_number), activation='softmax'))
        self.model.add(GaussianNoise(0.3))
        #self.model.add(Dropout(0.25))


        active = RMSprop()
        self.model.compile(loss='categorical_crossentropy', optimizer=active, metrics=['accuracy'])
        print(self.model.summary())


    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        self.create_network()
        #self.model.load(model_file)
        self.model.load_weights(model_file)
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        #(xs, ys) = numpy.reshape((xs, ys),(-1, self.string_to_number))
        self.create_network()
        self.model.fit(xs, ys, epochs=50, batch_size=64, validation_split=0.05)
        self.model.save(model_file)
        #score = self.model.evaluate(xs, ys)
        #print("\n%s: %.2f%%" % (self.model[1], score[1] * 100))

    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        x = numpy.array(x)
        x = x.reshape(-1, 86)
        #x = sequence.pad_sequences(x, maxlen=86, padding='post', value=0.33)
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
