import scipy.spatial as ss
import scipy.stats as sst
import numpy as np


class Similarity:
    @staticmethod
    def compute_similarity(feature, all_features, label_all, computation_number=3):
        """
        Get arousal and valence class from feature.
        Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 3 from each arousal and valence. 1 denotes low category, 2 denotes normal category, and 3 denotes high category.
        """

        sincerity_percentage = 0.97

        # Compute canberra with arousal training data
        #     distance_ar = map(lambda x: ss.distance.canberra(x, feature), all_features)
        distances = ss.distance.cdist(XA=[feature], XB=all_features,
                                      metric='canberra').reshape(-1)
        distances = list(distances)

        # Compute 3 nearest index and distance value from arousal
        idx_nearest = np.array(np.argsort(distances)[:computation_number])
        val_nearest = np.array(np.sort(distances)[:computation_number])

        result_ar = sst.mode(label_all[idx_nearest])
        result_ar = result_ar[0]

        return result_ar
