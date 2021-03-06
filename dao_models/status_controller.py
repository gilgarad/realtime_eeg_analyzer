""" status_controller.py: Controls the status of headset, analyze, ui linked variables """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"

import numpy as np


class StatusController:
    def __init__(self):
        """ Initialize the object for controlling status

        """
        # Headset connection status
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.channel_names = np.array(self.channel_names)
        self.electrodes_connection = dict()
        for channel in self.channel_names:
            self.electrodes_connection[channel] = 0 # 0 disconnected, 2 weak connection, 4 strong connection
        self.connection_status = 0
        self.disconnected_list = self.channel_names # disconnected at the beginning

        # Command status for Headset & Analysis
        self.analyze_status = 0 # 0 not analyze, 1 analyze
        self.headset_status = 0 # 0 not connected, 1 connection in process, 2 connected
        self.battery_level = 0

    def set_electrodes_connection(self, electrodes_status: list):
        # signal quality 0 None, 1 bad to 4 good
        cnt = 0
        disconnected_list = list()
        for idx, i in enumerate(electrodes_status):
            self.electrodes_connection[self.channel_names[idx]] = i

            if i > 0:
                cnt += 1
            else:
                disconnected_list.append(idx)

        self.disconnected_list = self.channel_names[disconnected_list]
        self.connection_status = int(float(cnt / 14) * 100)

    def set_battery_level(self, battery_level):
        self.battery_level = battery_level
