""" trail.py: DAO object that contains the trial information """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"

from datetime import datetime


class Trial:
    def __init__(self):
        """ Initialize Trial object

        """
        self.trial_name = ''
        self.game_name = ''
        self.survey_labels = dict()

        self.response_records = list()
        self.eeg_frequency = list()
        self.preanalyzed_values = list()
        self.connection_history = list()

    def store_preanalyzed_values(self, analyzed_values):
        """ Stores the emotiv's pre-analyzed values

        :param analyzed_values:
        :return:
        """
        new_data = [datetime.now()]
        new_data.append(analyzed_values['time'])
        new_data.extend(analyzed_values['met'])
        self.preanalyzed_values.append(new_data)

    def store_connection_history(self, electrode_status):
        """ Stores the connection history of a headset

        :param electrode_status:
        :return:
        """
        new_data = [datetime.now()]
        new_data.extend(electrode_status)
        self.connection_history.append(new_data)

    def store_fourier_transformed_frequency(self, frequency):
        """ Store fourier transformed features

        :param frequency:
        :return:
        """
        new_data = [datetime.now()]
        new_data.extend(frequency)
        self.eeg_frequency.append(new_data)

    def store_eeg_rawdata(self, rawdata):
        """ Store eeg rawdata

        :param rawdata:
        :return:
        """

        new_data = [datetime.now()]
        new_data.append(rawdata['time'])
        new_data.extend(rawdata['eeg'])
        self.response_records.append(new_data)

    def reset(self):
        """ Resets all the variables

        :return:
        """
        self.trial_name = ''
        self.game_name = ''
        self.survey_labels = dict()

        self.response_records = list()
        self.eeg_frequency = list()
        self.preanalyzed_values = list()
        self.connection_history = list()


