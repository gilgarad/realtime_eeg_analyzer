

class Vectorize:
    @staticmethod
    def vectorize(algorithm, all_data):
        features = list()

        for idx, data in enumerate(all_data):
            eeg_realtime = data.T

            feature = algorithm(eeg_realtime)
            features.append(feature)

        return features