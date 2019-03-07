import numpy as np
from datetime import datetime


class Trial:
    def __init__(self):
        self.trial_name = ''
        self.game_name = ''
        self.survey_labels = dict()

        self.response_records = list()
        self.eeg_frequency = list()
        self.preanalyzed_values = list()
        self.connection_history = list()

    def store_preanalyzed_values(self, analyzed_values):
        new_data = [datetime.now()]
        new_data.append(analyzed_values['time'])
        new_data.extend(analyzed_values['met'])
        self.preanalyzed_values.append(new_data)

    def store_connection_history(self, electrode_status):
        new_data = [datetime.now()]
        new_data.extend(electrode_status)
        self.connection_history.append(new_data)

    def store_fourier_transformed_frequency(self, frequency):
        new_data = [datetime.now()]
        new_data.extend(frequency)
        self.eeg_frequency.append(new_data)

    def store_eeg_rawdata(self, rawdata):
        new_data = [datetime.now()]
        new_data.append(rawdata['time'])
        new_data.extend(rawdata['eeg'])
        self.response_records.append(new_data)

    def reset(self):
        self.trial_name = ''
        self.game_name = ''
        self.survey_labels = dict()

        self.response_records = list()
        self.eeg_frequency = list()
        self.preanalyzed_values = list()
        self.connection_history = list()


