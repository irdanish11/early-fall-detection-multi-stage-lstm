import numpy as np
from random import shuffle
import os
import re


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


class CustomDataGenerator(object):

    def __init__(self,
                 data_path_context,
                 data_path_action,
                 batch_size=32,
                 temporal_length=30,
                 N_C=7):

        self.batch_size = batch_size
        self.data_path_action = data_path_action
        self.data_path_context = data_path_context
        self.temporal_length = temporal_length
        self.classes = N_C

        action_list = sorted(os.listdir(self.data_path_action),
                             key=natural_keys)
        features_action_list = list(
            filter(lambda x: 'feature' in x, action_list))
        self.features_action = [os.path.join(self.data_path_action, path) for
                                path in features_action_list]

        action_labels_list = list(filter(lambda x: 'label' in x, action_list))
        self.action_labels = [os.path.join(self.data_path_action, path) for path
                              in action_labels_list]

        context_list = sorted(os.listdir(self.data_path_context),
                              key=natural_keys)
        features_context_list = list(
            filter(lambda x: 'feature' in x, context_list))
        self.features_context = [os.path.join(self.data_path_context, path) for
                                 path in features_context_list]

        context_labels_list = list(filter(lambda x: 'label' in x, context_list))
        self.context_labels = [os.path.join(self.data_path_context, path) for
                               path in context_labels_list]

        self.pairs = list(zip(self.features_context, self.features_action,
                              self.context_labels, self.action_labels))
        shuffle(self.pairs)

        self.data_size = len(self.pairs)
        self.current = 0

    def generator(self):

        while True:

            if self.current < self.data_size - self.batch_size:

                X_c = np.zeros((self.batch_size, self.temporal_length, 1024))
                X_a = np.zeros((self.batch_size, self.temporal_length, 1024))
                y_c = np.zeros(
                    (self.batch_size, self.temporal_length, self.classes))
                y_a = np.zeros(
                    (self.batch_size, self.temporal_length, self.classes))

                cnt = 0
                for pair in range(self.current, self.current + self.batch_size):
                    X_c[cnt] = np.load(self.pairs[pair][0])
                    X_a[cnt] = np.load(self.pairs[pair][1])
                    y_c[cnt] = np.load(self.pairs[pair][2])
                    y_a[cnt] = np.load(self.pairs[pair][3])

                    cnt += 1

                yield (X_c, X_a), (y_c, y_a)

                self.current += self.batch_size

            else:

                self.current = 0
                shuffle(self.pairs)

                X_c = np.zeros((self.batch_size, self.temporal_length, 1024))
                X_a = np.zeros((self.batch_size, self.temporal_length, 1024))
                y_c = np.zeros(
                    (self.batch_size, self.temporal_length, self.classes))
                y_a = np.zeros(
                    (self.batch_size, self.temporal_length, self.classes))

                cnt = 0
                for pair in range(self.current, self.current + self.batch_size):
                    X_c[cnt] = np.load(self.pairs[pair][0])
                    X_a[cnt] = np.load(self.pairs[pair][1])
                    y_c[cnt] = np.load(self.pairs[pair][2])
                    y_a[cnt] = np.load(self.pairs[pair][3])
                    cnt += 1

                yield (X_c, X_a), (y_c, y_a)

                self.current += self.batch_size


'''
how to use:
train_generator = CustomDataGenerator(*params1).generator()
validation_generator = CustomDataGenerator(*params2).generator()

model.fit_generator(generator = training_generator,
                    validation_data = validation_generator,
                    nb_epoch = 50,
                    verbose = 1)

'''
