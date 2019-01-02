import scipy.spatial as ss
import numpy as np
import collections


class Similarity:
    @staticmethod
    def compute_similarity(feature, all_features, label_all, computation_number=3):
        """
        Get arousal and valence class from feature.
        Input: Feature (standard deviasion and mean) from all frequency bands and channels with dimesion 1 x M (number of feature).
        Output: Class of emotion between 1 to 3 from each arousal and valence. 1 denotes low category, 2 denotes normal category, and 3 denotes high category.
        """
        # Compute canberra with arousal training data
        #     distance_ar = map(lambda x: ss.distance.canberra(x, feature), all_features)
        distances = ss.distance.cdist(XA=[feature], XB=all_features,
                                      metric='canberra').reshape(-1)
        distances = list(distances)

        # Compute 3 nearest index and distance value from arousal
        idx_nearest = np.array(np.argsort(distances)[:computation_number])
        val_nearest = np.array(np.sort(distances)[:computation_number])

        a = collections.Counter(label_all[idx_nearest]).most_common(2)

        # If first and second count same then more examples will be used
        found = False
        while not found:
            if len(a) != 1 and a[0][1] == a[1][1]:
                computation_number += 2
                idx_nearest = np.array(np.argsort(distances)[:computation_number])
                val_nearest = np.array(np.sort(distances)[:computation_number])

                a = collections.Counter(label_all[idx_nearest]).most_common(2)
            else:
                found = True

        most_common_similarity = a[0][0]

        return most_common_similarity

