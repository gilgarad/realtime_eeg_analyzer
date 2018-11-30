import numpy as np
import scipy.spatial as ss
import scipy.stats as sst
import csv
import sys

# from os.path import join, dirname
#
# sys.path.append(join(join(dirname(__file__), '..'), 'utils'))

from utils.similarity import Similarity
from utils.vectorize import Vectorize


class FFTConvention:
    def __init__(self, path):
        self.train_arousal = self.get_csv(path + "train_arousal.csv")
        self.train_valence = self.get_csv(path + "train_valence.csv")
        self.class_arousal = self.get_csv(path + "class_arousal.csv")
        self.class_valence = self.get_csv(path + "class_valence.csv")

    def get_csv(self, path):
        """
        Get data from csv and convert them to numpy python.
        Input: Path csv file.
        Output: Numpy array from csv data.
        """
        # Get csv data to list
        file_csv = open(path)
        data_csv = csv.reader(file_csv)
        data_training = np.array([each_line for each_line in data_csv])

        # Convert list to float
        data_training = data_training.astype(np.double)

        return data_training

    def do_fft(self, all_channel_data):
        """
        Do fft in each channel for all channels.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
        """
        data_fft = map(lambda x: np.fft.fft(x), all_channel_data)

        return data_fft

    def get_frequency(self, all_channel_data):
        """
        Get frequency from computed fft for all channels.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
        """
        # Length each data channel
        L = len(all_channel_data[0])
        # print(L)

        # Sampling frequency
        Fs = 128

        # Get fft data
        data_fft = self.do_fft(all_channel_data)
        # print(data_fft)

        # Compute frequency
        frequency = map(lambda x: abs(x / L), data_fft)
        frequency = map(lambda x: x[: int(L / 2) + 1] * 2, frequency)
        frequency = list(frequency)

        # List frequency
        delta = map(lambda x: x[int(L * 1 / Fs) - 1: int(L * 4 / Fs)], frequency)
        theta = map(lambda x: x[int(L * 4 / Fs) - 1: int(L * 8 / Fs)], frequency)
        alpha = map(lambda x: x[int(L * 5 / Fs) - 1: int(L * 13 / Fs)], frequency)
        beta = map(lambda x: x[int(L * 13 / Fs) - 1: int(L * 30 / Fs)], frequency)
        gamma = map(lambda x: x[int(L * 30 / Fs) - 1: int(L * 50 / Fs)], frequency)

        # print('int(L * 4 / Fs)', int(L * 4 / Fs))

        return delta, theta, alpha, beta, gamma

    def get_feature(self, all_channel_data):
        """
        Get feature from each frequency.
        Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
        Output: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        """
        # Get frequency data
        (delta, theta, alpha, beta, gamma) = self.get_frequency(all_channel_data)
        delta = list(delta)
        theta = list(theta)
        alpha = list(alpha)
        beta = list(beta)
        gamma = list(gamma)
        # print(delta)


        # Compute feature std
        delta_std = np.std(delta, axis=1)
        theta_std = np.std(theta, axis=1)
        alpha_std = np.std(alpha, axis=1)
        beta_std = np.std(beta, axis=1)
        gamma_std = np.std(gamma, axis=1)

        # Compute feature mean
        delta_m = np.mean(delta, axis=1)
        theta_m = np.mean(theta, axis=1)
        alpha_m = np.mean(alpha, axis=1)
        beta_m = np.mean(beta, axis=1)
        gamma_m = np.mean(gamma, axis=1)

        # Concate feature
        feature = np.array(
            [delta_std, delta_m, theta_std, theta_m, alpha_std, alpha_m, beta_std, beta_m, gamma_std, gamma_m])
        feature = feature.T
        feature = feature.ravel()

        return feature

    def predict_emotion(self, feature):
        """
        Get arousal and valence class from feature.
        Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 3 from each arousal and valence. 1 denotes low category, 2 denotes normal category, and 3 denotes high category.
        """

        computation_number = 3
        sincerity_percentage = 0.97

        # Compute canberra with arousal training data
        distance_ar = map(lambda x: ss.distance.canberra(x, feature), self.train_arousal)
        # distance_ar = map(lambda x: ss.distance.cosine(x, feature), self.train_arousal)
        distance_ar = list(distance_ar)
        # print(distance_ar)

        # Compute canberra with valence training data
        distance_va = map(lambda x: ss.distance.canberra(x, feature), self.train_valence)
        # distance_va = map(lambda x: ss.distance.cosine(x, feature), self.train_valence)
        distance_va = list(distance_va)
        # print(distance_va)

        # Compute 3 nearest index and distance value from arousal
        idx_nearest_ar = np.array(np.argsort(distance_ar)[:computation_number])
        val_nearest_ar = np.array(np.sort(distance_ar)[:computation_number])

        # Compute 3 nearest index and distance value from arousal
        idx_nearest_va = np.array(np.argsort(distance_va)[:computation_number])
        val_nearest_va = np.array(np.sort(distance_va)[:computation_number])
        # TODO DO IT FROM HERE
        # concern sst.mode picks count max, find "distance_va idx" with "that count max" not the idx=0
        # distance_va[idx_nearest_va[0]]

        # Compute comparation from first nearest and second nearest distance. If comparation less or equal than 0.7, then take class from the first nearest distance. Else take frequently class.
        # Arousal
        comp_ar = val_nearest_ar[0] / val_nearest_ar[1]
        if comp_ar <= sincerity_percentage:
            result_ar = self.class_arousal[0, idx_nearest_ar[0]]
        else:
            result_ar = sst.mode(self.class_arousal[0, idx_nearest_ar])
            result_ar = float(result_ar[0])
            # print(result_ar)
        # result_ar = sst.mode(self.class_arousal[0, idx_nearest_ar])
        # result_ar = float(result_ar[0])

        # Valence
        comp_va = val_nearest_va[0] / val_nearest_va[1]
        if comp_va <= sincerity_percentage:
            # print('va <= 0.97')
            result_va = self.class_valence[0, idx_nearest_va[0]]
        else:
            result_va = sst.mode(self.class_valence[0, idx_nearest_va])
            result_va = float(result_va[0])
        # result_va = sst.mode(self.class_valence[0, idx_nearest_va])
        # result_va = float(result_va[0])
        # print(result_ar, result_va)
        return result_ar, result_va

    def determine_emotion_class(self, feature):
        """
        Get emotion class from feature.
        Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 5 according to Russel's Circumplex Model.
        """
        class_ar, class_va = self.predict_emotion(feature)

        if class_ar == 2.0 or class_va == 2.0:
            emotion_class = 5 # neutral
        elif class_ar == 3.0 and class_va == 1.0:
            emotion_class = 1 # fear - nervous - stress - tense - upset
        elif class_ar == 3.0 and class_va == 3.0:
            emotion_class = 2 # happy - alert - excited - elated
        elif class_ar == 1.0 and class_va == 3.0:
            emotion_class = 3 # relax - calm - serene - contented
        elif class_ar == 1.0 and class_va == 1.0:
            emotion_class = 4 # sad - depressed - lethargic - fatigue

        return emotion_class

    def get_emotion(self, all_channel_data):
        # Get feature from EEG data
        feature = self.get_feature(all_channel_data)

        # Predict emotion class
        emotion_class = self.determine_emotion_class(feature)

        return emotion_class

    def std_test(self, x_train, y_train, x_test, y_test):
        print('##########')
        print('STD Test')
        print('##########')

        # _x_train = self.std_vectorize(all_data=x_train)
        # _x_test = self.std_vectorize(all_data=x_test)
        _x_train = Vectorize.vectorize(algorithm=self.get_feature, all_data=x_train)
        _x_test = Vectorize.vectorize(algorithm=self.get_feature, all_data=x_test)

        count = 0

        for idx, (x, y) in enumerate(zip(_x_test, y_test)):
            _y = Similarity.compute_similarity(feature=x, all_features=_x_train, label_all=y_train, computation_number=5)
            #     print(idx, y, _y)
            if y == _y[0]:
                count += 1

        print('##########')
        print("STD Similarity Percentage: %.4f" % (float(count / (idx + 1))))
        print('##########')

        # return _x_train, _x_test, y_train, y_test

        # with K.tf.device('/gpu:0'):
        #     model = Sequential()
        #
        #     model.add(Dense(64, activation='sigmoid'))
        #     model.add(Dense(64, activation='sigmoid'))
        #     model.add(Flatten())
        #     model.add(Dense(64, activation='sigmoid'))
        #     model.add(Dense(1, activation='sigmoid'))
        #
        #     adam = optimizers.Adam()
        #     model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
        #
        # model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)