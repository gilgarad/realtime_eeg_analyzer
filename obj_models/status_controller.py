class StatusController:
    def __init__(self):
        # self.headset_connection = 0 # signal quality
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.electrodes_connection = dict()
        for channel in self.channel_names:
            self.electrodes_connection[channel] = 0 # 0 disconnected, 2 weak connection, 4 strong connection

        self.analyze_status = False # 0 not analyze, 1 analyze
        self.headset_status = False # 0 not connected, 1 connection in process, 2 connected