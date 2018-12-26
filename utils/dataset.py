import numpy as np
from os import listdir
from os.path import isdir, isfile, join
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


class Dataset:
    @staticmethod
    def scaler(data_path, folder):
        files = [f for f in listdir(join(data_path, folder))]
        data = np.concatenate([np.load(join(data_path, folder, file)) for file in files], axis=0)
        data = data[:,5:-2]

        sc = MinMaxScaler()
        sc.fit(data)

        return sc

    @staticmethod
    def make_sequence_data(data, frequency=128, data_length_in_time=5, sliding_window_in_time=5, local_scaling=False):
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
        # labels
        labels = ['안정기', '전투', '휴식']
        data_dict = dict()

        number_of_channels = 14

        train_data = np.empty(shape=(0, frequency * data_length_in_time, number_of_channels))
        train_labels = np.empty(shape=(0))
        test_data = np.empty(shape=(0, frequency * data_length_in_time, number_of_channels))
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



