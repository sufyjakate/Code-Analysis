import tensorflow as tf
import numpy
import tflearn
from keras import Input
from keras.layers import Dense, Dropout, regularizers, Embedding, LSTM, Bidirectional, TimeDistributed, Flatten, \
    GlobalAveragePooling1D
from keras.models import Sequential
from keras.optimizers import SGD, Adagrad, RMSprop
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

        print("x,y pairs: " + str(len(xs)))
        (xs, ys) = sequence.pad_sequences((xs, ys), maxlen=len(self.string_to_number))
        #(xs, ys) = (xs, ys).reshape([-1, 28, 28, 1])
        #(xs, ys) = Input(shape=(xs, ys).shape, name=(xs, ys))
        return (xs, ys)

    def create_network(self):
        self.model = Sequential()
        self.model.add(Embedding(5000, 32, input_length=len(self.string_to_number)))
        #self.model.add(Bidirectional(LSTM(100, dropout=0.8, return_sequences=True)))
        #self.model.add(Flatten())
        self.model.add(GlobalAveragePooling1D())
        #self.model.add(LSTM(len(self.string_to_number), return_sequences=False))
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(Dense(len(self.string_to_number), activation='linear'))
        opt = RMSprop()
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
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
        self.model.fit(xs, ys, epochs=1000, batch_size=258, validation_split=0.05)
        self.model.save(model_file)
        #score = self.model.evaluate(xs, ys)
        #print("\n%s: %.2f%%" % (self.model[1], score[1] * 100))

    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        x = numpy.array(x)
        x = x.reshape(-1, 86)
        x = sequence.pad_sequences(x, maxlen=len(self.string_to_number))
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist()
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        best_token = self.string_to_token(best_string)
        return [best_token]
