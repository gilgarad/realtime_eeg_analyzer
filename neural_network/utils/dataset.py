""" custom_function.py: Keras custom defined functions placed in this file """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"

import numpy as np
from os import listdir
from os.path import isdir, isfile, join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# newly added
from keras.preprocessing.sequence import pad_sequences


class Dataset:
    """
    Dataset for EEG data
    """

    ########## new ##########
    def __init__(self, data_path, num_channels=14, max_minutes=10, num_original_features=18, num_reduced_features=10,
                 augment=False, stride=128, delete_range=128, data_status='rawdata', sampling_rate=128, padding='all'):
        """ Initialize Dataset object

        :param data_path:
        :param num_channels:
        :param max_minutes:
        :param num_original_features:
        :param num_reduced_features:
        :param augment:
        :param stride:
        :param delete_range:
        :param data_status:
        """
        self.data_path = data_path
        self.num_channels = num_channels
        self.max_minutes = max_minutes
        self.num_original_features = num_original_features
        self.num_reduced_features = num_reduced_features
        self.augment = augment
        self.stride = stride
        self.delete_range = delete_range
        self.data_status = data_status

        self.sequence_length = 0
        self.padding = padding
        # Temp
        self.sampling_rate = sampling_rate

        self.data_dict = self.load_make_data(data_path=self.data_path, augment=self.augment, stride=self.stride,
                                             delete_range=self.delete_range, data_status=self.data_status,
                                             padding=self.padding)

    def load_make_data(self, data_path, augment=False, stride=128, delete_range=128, data_status='rawdata',
                       padding='all'):
        """ Load all data in the given data_path and place them by its file name in a dictionary

        :param data_path:
        :param augment:
        :param stride:
        :param delete_range:
        :param data_status:
        :param padding: pre, post, all
        :return:
        """
        print('Loading Data')

        data_dict = dict()
        cnt = 0
        max_sequence_length = 0

        padding_dict = {
            'all': ['post', 'pre'],
            'pre': ['pre'],
            'post': ['post']
        }

        for fname in listdir(data_path):
            if 'npy' not in fname:
                #             print(fname)
                continue

            data_name = fname.replace('.npy', '')
            if data_name in data_dict:
                continue

            cnt += 1

            print(cnt, data_name)

            if data_status == 'rawdata':
                self.sequence_length = 128 * 60 * self.max_minutes

                data = np.load(join(data_path, fname)).tolist()

                input_data = data['eeg'][:, 5:-2]  # slice unnecessary features
                if max_sequence_length < input_data.shape[0]:
                    max_sequence_length = input_data.shape[0]

                if augment:
                    input_data = self.augment_data(input_data, stride=stride, delete_range=delete_range)
                else:
                    input_data = input_data.reshape(1, input_data.shape[0], input_data.shape[1])

                # pad pre and post
                labels = data['labels']
                labels_fun = labels['amusement']
                labels_immersion = labels['immersion']
                labels_difficulty = labels['difficulty']
                labels_emotion = labels['emotion']

                target_data = np.empty(shape=(0, 4))
                all_input_list = list()
                for pad_type in padding_dict[padding]:
                    input_data_pad = pad_sequences(input_data, maxlen=self.sequence_length, padding=pad_type)
                    all_input_list.append(input_data_pad)
                    target_data = np.concatenate([target_data,
                                                  [[labels_fun, labels_immersion, labels_difficulty, labels_emotion]] \
                                                  * input_data_pad.shape[0]], axis=0)

                target_data = target_data.T

                data_dict[data_name] = [np.concatenate(all_input_list, axis=0), target_data]

            elif data_status == 'fourier_transform':
                print('')
            elif data_status == 'pre_fourier_transformed':
                self.sequence_length = 60 * self.max_minutes

                data = np.load(join(data_path, fname))
                # input_data = data[:, 0][:self.sampling_rate].tolist()
                # 0~127 window=1 tick=1 shifted, 128~255 window=2 tick=2 shifted and so on
                input_data = data[::self.sampling_rate, 0].tolist()
                input_data = np.array(input_data)
                input_data = input_data.reshape(input_data.shape[0], self.sequence_length,
                                                self.num_channels * self.num_original_features)
                if max_sequence_length < input_data.shape[1]:
                    max_sequence_length = input_data.shape[1]

                # target_data = data[:, 1][:self.sampling_rate].tolist()
                target_data = data[::self.sampling_rate, 1].tolist()
                target_data = np.array(target_data)
                target_data = target_data.transpose(1, 0, 2).reshape(4, target_data.shape[0])

                #         x_train = np.concatenate([x_train, _x_train], axis=0)
                #         y_train = np.concatenate([y_train, _y_train], axis=1)

                data_dict[data_name] = [input_data, target_data]

        print('max sequence length: %i' % max_sequence_length)
        return data_dict

    def augment_data(self, input_data, delete_range=128, stride=128):
        """ Augment data, currently only augment by delete position in every stride, and pad zero 'pre' and 'post'

        :param input_data:
        :param delete_range:
        :param stride:
        :return:
        """
        augmented_input_data = np.empty(shape=(
        0, (int(input_data.shape[0] / stride) + 1) * stride - (int(input_data.shape[0] / stride) + 1),
        input_data.shape[1]))
        #     print(augmented_input_data.shape)

        for i in range(delete_range):
            #         print(i)
            new_input_data = np.copy(input_data)
            new_input_data = new_input_data.reshape(1, new_input_data.shape[0], new_input_data.shape[1])
            new_input_data = pad_sequences(new_input_data, maxlen=(int(input_data.shape[0] / stride) + 1) * stride,
                                           padding='post')
            #         new_input_data = new_input_data.reshape(new_input_data.shape[1], new_input_data.shape[2])
            new_input_data = new_input_data.reshape(int(input_data.shape[0] / stride) + 1, stride,
                                                    new_input_data.shape[2])
            new_input_data = np.delete(new_input_data, i, axis=1)
            new_input_data = new_input_data.reshape(1, new_input_data.shape[0] * (stride - 1),
                                                    new_input_data.shape[2]).astype(float)
            #         print(new_input_data.shape)
            augmented_input_data = np.concatenate([augmented_input_data, new_input_data], axis=0)

        return augmented_input_data

    def get_data(self, data_dict, train_names, test_names, feature_type='pre_fourier_transformed',
                 is_classification=True):
        """ Get datasets composed of x_train, y_train, x_valid, y_valid

        :param data_dict:
        :param train_names:
        :param test_names:
        :param feature_type:
        :param is_classification:
        :return:
        """

        label_dict = {
            'amusement': 0,
            'immersion': 1,
            'difficulty': 2,
            'emotion': 3
        }

        if feature_type == 'all':
            x_train = np.empty(shape=(0, self.sequence_length, self.num_channels * self.num_original_features))
            x_test = np.empty(shape=(0, self.sequence_length, self.num_channels * self.num_original_features))
        elif feature_type == 'pre_fourier_transformed':
            x_train = np.empty(shape=(0, self.sequence_length, self.num_channels * self.num_reduced_features))
            x_test = np.empty(shape=(0, self.sequence_length, self.num_channels * self.num_reduced_features))
        elif feature_type == 'rawdata':
            x_train = np.empty(shape=(0, self.sequence_length, self.num_channels))
            x_test = np.empty(shape=(0, self.sequence_length, self.num_channels))

        y_train = np.empty(shape=(4, 0))
        y_test = np.empty(shape=(4, 0))

        all_data_list = list()
        all_labels_list = list()
        all_data_list.append(x_train)
        all_labels_list.append(y_train)
        for data_name in train_names:
            all_data, all_labels = data_dict[data_name]
            if feature_type == 'pre_fourier_transformed':
                all_data = all_data.reshape((all_data.shape[0], all_data.shape[1], self.num_channels, self.num_original_features))
                all_data = all_data[:, :, :, :self.num_reduced_features]
                all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], self.num_channels * self.num_reduced_features)

            all_data_list.append(all_data)
            all_labels_list.append(all_labels)

        x_train = np.concatenate(all_data_list, axis=0)
        y_train = np.concatenate(all_labels_list, axis=1)

        print('x_train.shape', x_train.shape)
        print('y_train[0].shape', y_train[0].shape)

        all_data_list = list()
        all_labels_list = list()
        all_data_list.append(x_test)
        all_labels_list.append(y_test)
        for data_name in test_names:
            all_data, all_labels = data_dict[data_name]
            if feature_type == 'pre_fourier_transformed':
                all_data = all_data.reshape((all_data.shape[0], all_data.shape[1], self.num_channels, self.num_original_features))
                all_data = all_data[:, :, :, :self.num_reduced_features]
                all_data = all_data.reshape(all_data.shape[0], all_data.shape[1], self.num_channels * self.num_reduced_features)

            all_data_list.append(all_data)
            all_labels_list.append(all_labels)

        x_test = np.concatenate(all_data_list, axis=0)
        y_test = np.concatenate(all_labels_list, axis=1)

        _y_train = y_train
        _y_test = y_test

        __y_train = list()
        __y_test = list()

        if is_classification:
            # To make classification problem, convert scores to class
            for idx, (label_name, label_idx) in enumerate(label_dict.items()):
                print('')
                print('Label:', label_name)
                y_train = _y_train[label_idx]
                y_test = _y_test[label_idx]

                if label_name == 'amusement':
                    y_train = np.round(y_train.astype(float))
                    y_test = np.round(y_test.astype(float))
                    y_train = y_train.astype(int)
                    y_test = y_test.astype(int)

                    median_num = np.median(np.concatenate([y_train, y_test], axis=0))
                    print('Median num:', median_num)

                    indices = np.where(y_train < median_num)
                    y_train[indices] = 0
                    indices = np.where(y_test < median_num)
                    y_test[indices] = 0
                    indices = np.where(y_train == median_num)
                    y_train[indices] = 0
                    indices = np.where(y_test == median_num)
                    y_test[indices] = 0
                    indices = np.where(y_train > median_num)
                    y_train[indices] = 1
                    indices = np.where(y_test > median_num)
                    y_test[indices] = 1

                elif label_name == 'immersion':
                    y_train = np.round(y_train.astype(float))
                    y_test = np.round(y_test.astype(float))
                    y_train = y_train.astype(int)
                    y_test = y_test.astype(int)

                    median_num = np.median(np.concatenate([y_train, y_test], axis=0))
                    print('Median num:', median_num)

                    indices = np.where(y_train <= median_num)
                    y_train[indices] = 0
                    indices = np.where(y_test <= median_num)
                    y_test[indices] = 0
                    indices = np.where(y_train > median_num)
                    y_train[indices] = 1
                    indices = np.where(y_test > median_num)
                    y_test[indices] = 1
                elif label_name == 'difficulty':
                    y_train = np.round(y_train.astype(float))
                    y_test = np.round(y_test.astype(float))
                    y_train = y_train.astype(int)
                    y_test = y_test.astype(int)

                    median_num = np.median(np.concatenate([y_train, y_test], axis=0))
                    print('Median num:', median_num)

                    indices = np.where(y_train <= median_num)
                    y_train[indices] = 0
                    indices = np.where(y_test <= median_num)
                    y_test[indices] = 0
                    indices = np.where(y_train > median_num)
                    y_train[indices] = 1
                    indices = np.where(y_test > median_num)
                    y_test[indices] = 1
                elif label_name == 'emotion':
                    # y_train = np.core.defchararray.strip(y_train)
                    # y_test = np.core.defchararray.strip(y_test)
                    # indices = np.where(np.core.defchararray.find(y_train, 'happy') == 0)
                    # y_train[indices] = 0
                    # indices = np.where(np.core.defchararray.find(y_test, 'happy') == 0)
                    # y_test[indices] = 0
                    # indices = np.where(np.core.defchararray.find(y_train, 'neutral') == 0)
                    # y_train[indices] = 1
                    # indices = np.where(np.core.defchararray.find(y_test, 'neutral') == 0)
                    # y_test[indices] = 1
                    # indices = np.where(np.core.defchararray.find(y_train, 'annoyed') == 0)
                    # y_train[indices] = 2
                    # indices = np.where(np.core.defchararray.find(y_test, 'annoyed') == 0)
                    # y_test[indices] = 2
                    #
                    # y_train = y_train.astype(int)
                    # y_test = y_test.astype(int)

                    y_train = np.round(y_train.astype(float))
                    y_test = np.round(y_test.astype(float))
                    y_train = y_train.astype(int)
                    y_test = y_test.astype(int)

                    indices = np.where(y_train <= 3)
                    y_train[indices] = 2
                    indices = np.where(y_test <= 3)
                    y_test[indices] = 2
                    indices = np.where((y_train > 3) & (y_train <= 6))
                    y_train[indices] = 1
                    indices = np.where((y_test > 3) & (y_test <= 6))
                    y_test[indices] = 1
                    indices = np.where(y_train > 6)
                    y_train[indices] = 0
                    indices = np.where(y_test > 6)
                    y_test[indices] = 0

                unique_labels = np.unique(y_train, axis=0)
                print('Unique Labels:', unique_labels)
                print('In Train')
                for i in unique_labels:
                    print('class %i has %i' % (i, len(np.where(y_train == i)[0])))

                print('In Test')
                for i in unique_labels:
                    print('class %i has %i' % (i, len(np.where(y_test == i)[0])))

                #         y_train = np_utils.to_categorical(y_train)
                #         y_test = np_utils.to_categorical(y_test)

                __y_train.append(y_train)
                __y_test.append(y_test)

            #         if idx == 3:
            #             break

            #     x_train, y_train = shuffle(x_train, y_train)

            y_train = np.array(__y_train)
            y_test = np.array(__y_test)

        return [x_train, y_train, x_test, y_test]

    ########## old ##########

    @staticmethod
    def scaler(data_path, folder):
        """ ???

        :param data_path:
        :param folder:
        :return:
        """
        files = [f for f in listdir(join(data_path, folder))]
        data = np.concatenate([np.load(join(data_path, folder, file)) for file in files], axis=0)
        data = data[:,5:-2]

        sc = MinMaxScaler()
        sc.fit(data)

        return sc

    @staticmethod
    def make_sequence_data(data, frequency=128, data_length_in_time=5, sliding_window_in_time=5, local_scaling=False):
        """

        :param data:
        :param frequency:
        :param data_length_in_time:
        :param sliding_window_in_time:
        :param local_scaling:
        :return:
        """
        data_length = int(frequency * data_length_in_time)
        sliding_size = int(frequency * sliding_window_in_time)

        if local_scaling:
            sc = MinMaxScaler()

        sequence_data = np.empty(shape=(0, data_length, 14))
        for idx in range(0, len(data), sliding_size):
            d = data[idx: idx + data_length]
            if local_scaling:
                d = sc.fit_transform(d)
            if len(d) < data_length:
                break
            #         print(d)
            sequence_data = np.concatenate([sequence_data, [d]], axis=0)

        return sequence_data

    @staticmethod
    def make_dataset_battle_rest(data_path, frequency=128, data_length_in_time=1, sliding_window_in_time=1,
                                 augment_length=False, train_test_ratio=0.2, train_names=list(), test_names=list(),
                                 global_scaling=True, local_scaling=False, remove_label_index=list()):
        """ Builds up the dataset for battle / rest(non-battle) status lineage revolution data

        :param data_path:
        :param frequency:
        :param data_length_in_time:
        :param sliding_window_in_time:
        :param augment_length:
        :param train_test_ratio:
        :param train_names:
        :param test_names:
        :param global_scaling:
        :param local_scaling:
        :param remove_label_index:
        :return:
        """
        # labels
        labels = ['안정기', '전투', '휴식']
        data_dict = dict()

        number_of_channels = 14

        train_data = np.empty(shape=(0, int(frequency * data_length_in_time), number_of_channels))
        train_labels = np.empty(shape=(0))
        test_data = np.empty(shape=(0, int(frequency * data_length_in_time), number_of_channels))
        test_labels = np.empty(shape=(0))

        # opath 를 제외한 모든 하위 디렉토리에서 데이터를 읽은 뒤 opath에 하나의 label로 저장
        for folder_name in listdir(data_path):
            if not isdir(join(data_path, folder_name)):
                continue

            if folder_name not in train_names and folder_name not in test_names:
                continue

            #         print(folder_name)
            if global_scaling:
                sc = Dataset.scaler(data_path, folder_name)

            for fname in listdir(join(data_path, folder_name)):

                if not isfile(join(data_path, folder_name, fname)):
                    continue

                label_idx = -1
                for idx, _name in enumerate(labels):
                    if _name in fname:
                        label_idx = idx
                        break

                if label_idx == -1:
                    print('Wrong label in the file list:', fname)
                    continue

                data = np.load(join(data_path, folder_name, fname))
                data = data[:, 5:-2]  # slice unnecessary features

                if global_scaling:
                    data = sc.transform(data)
                sequence_data = Dataset.make_sequence_data(data, frequency=frequency,
                                                           data_length_in_time=data_length_in_time,
                                                           sliding_window_in_time=sliding_window_in_time)

                if folder_name in train_names:
                    train_data = np.concatenate([train_data, sequence_data], axis=0)
                    train_labels = np.concatenate([train_labels, [label_idx] * len(sequence_data)], axis=0)
                elif folder_name in test_names:
                    test_data = np.concatenate([test_data, sequence_data], axis=0)
                    test_labels = np.concatenate([test_labels, [label_idx] * len(sequence_data)], axis=0)

        if test_data.shape[0] == 0:
            print('No test data provided. Splits with ratio: %.2f' % (train_test_ratio))
            train_data, test_data, train_labels, test_labels = train_test_split(train_data, train_labels,
                                                                                test_size=train_test_ratio,
                                                                                random_state=777)

        data_size = list()
        max_length = 0
        for i in range(len(labels)):
            data_index = np.argwhere(train_labels == i)
            if len(data_index) > max_length:
                max_length = len(data_index)
            data_size.append(data_index)

        if augment_length:
            print('Data augmented with original shape')
            print('Train data shape:', train_data.shape)
            print('Train labels shape:', train_labels.shape)
            print('Stable: %i Battle: %i Rest: %i' % (len(data_size[0]), len(data_size[1]), len(data_size[2])))

            _data_size = list()
            n_train_data = train_data
            n_train_labels = train_labels

            for idx, data_index in enumerate(data_size):
                data_to_augment = int(max_length / len(data_index)) - 1
                #             print('data_to_augment', data_to_augment)
                n_data_index = data_index

                for i in range(data_to_augment):
                    data_index = np.concatenate([data_index, n_data_index], axis=0)

                left_over = max_length % len(data_index)
                if left_over != 0:
                    data_index = np.concatenate([data_index, data_index[:left_over]], axis=0)

                data_index = data_index.ravel()

                if idx == 0:
                    train_data = n_train_data[data_index]
                    train_labels = n_train_labels[data_index]
                else:
                    #                 print(len(n_train_data))
                    train_data = np.concatenate([train_data, n_train_data[data_index]], axis=0)
                    train_labels = np.concatenate([train_labels, n_train_labels[data_index]], axis=0)

                _data_size.append(data_index)

            data_size = _data_size
            print('Stable: %i Battle: %i Rest: %i' % (len(data_size[0]), len(data_size[1]), len(data_size[2])))
            print('')

        train_data, train_labels = shuffle(train_data, train_labels)

        for i in remove_label_index:
            train_index = np.argwhere(train_labels == i)
            test_index = np.argwhere(test_labels == i)
            train_index = train_index.ravel()
            test_index = test_index.ravel()

            #     print(len(train_index))
            #     print(len(test_index))

            train_data = np.delete(train_data, train_index, axis=0)
            train_labels = np.delete(train_labels, train_index)
            test_data = np.delete(test_data, test_index, axis=0)
            test_labels = np.delete(test_labels, test_index)

        print('Train data shape:', train_data.shape)
        print('Train labels shape:', train_labels.shape)
        print('Battle: %i Rest: %i' % (len(data_size[1]), len(data_size[2])))
        #     print('Stable: %i Battle: %i Rest: %i' % ( len(data_size[0]), len(data_size[1]), len(data_size[2])))

        print('Test data shape:', test_data.shape)
        print('Test labels shape:', test_labels.shape)
        print('Battle: %i Rest: %i' % (len(test_labels[test_labels == 1]), len(test_labels[test_labels == 2])))
        #     print('Stable: %i Battle: %i Rest: %i' % ( len(data_size[0]), len(data_size[1]), len(data_size[2])))

        return train_data, train_labels, test_data, test_labels


    @staticmethod
    def augment_data_old(x, y):
        """
        Make all classes in dataset to be same by augmenting the smaller ones to be the largest one

        :param x:
        :param y:
        :return:
        """
        #     if augment_length:
        unique_labels = np.unique(y)
        print(unique_labels)

        max_length = 0
        for label in unique_labels:
            if max_length < len(np.where(y == label)[0]):
                max_length = len(np.where(y == label)[0])

        n_train_data = x
        n_train_labels = y

        for idx, label in enumerate(unique_labels):
            print('label', label)
            data_index = np.where(n_train_labels == label)[0]
            print(len(data_index))
            data_to_augment = int(max_length / len(data_index) - 1)
            #             print('data_to_augment', data_to_augment)

            n_data_index = data_index

            for i in range(data_to_augment):
                data_index = np.concatenate([data_index, n_data_index], axis=0)

            left_over = max_length % len(data_index)
            if left_over != 0:
                data_index = np.concatenate([data_index, n_data_index[:left_over]], axis=0)

            #         data_index = data_index.ravel()

            if idx == 0:
                x = n_train_data[data_index]
                y = n_train_labels[data_index]
            else:
                #                 print(len(n_train_data))
                x = np.concatenate([x, n_train_data[data_index]], axis=0)
                y = np.concatenate([y, n_train_labels[data_index]], axis=0)

        return x, y
