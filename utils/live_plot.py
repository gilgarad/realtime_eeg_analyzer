# copied from: https://stackoverflow.com/questions/11874767/how-do-i-plot-in-real-time-in-a-while-loop-using-matplotlib

###################################################################
#                                                                 #
#                     PLOTTING A LIVE GRAPH                       #
#                  ----------------------------                   #
#            EMBED A MATPLOTLIB ANIMATION INSIDE YOUR             #
#            OWN GUI!                                             #
#                                                                 #
###################################################################


import sys
import os
from PyQt5 import QtGui, QtWidgets
from PyQt5 import QtCore
import functools
import numpy as np
import random as rd
import matplotlib
matplotlib.use("Qt4Agg")
from matplotlib.figure import Figure
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import time
import threading
from datetime import datetime
from os.path import join, dirname
image_path = join(join(dirname(__file__), '..'), 'webapp', 'public', 'img')



def setCustomSize(x, width, height):
    sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    sizePolicy.setHeightForWidth(x.sizePolicy().hasHeightForWidth())
    x.setSizePolicy(sizePolicy)
    x.setMinimumSize(QtCore.QSize(width, height))
    x.setMaximumSize(QtCore.QSize(width, height))

''''''


class CustomMainWindow(QtWidgets.QMainWindow):

    def __init__(self):

        super(CustomMainWindow, self).__init__()
        umulti = 100 # unit multiplier

        # Define the geometry of the main window
        self.setGeometry(3 * umulti, 1 * umulti, 12 * umulti, 8 * umulti) # x, y, width, height
        self.setWindowTitle("Realtime EEG Analysis")

        # Create FRAME_A
        self.FRAME_A = QtWidgets.QFrame(self)
        # self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(210, 210, 235, 255).name())
        # self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(214, 240, 255, 255).name())
        self.FRAME_A.setStyleSheet("QWidget { background-color: %s }" % QtGui.QColor(218, 217, 255, 255).name())
        self.LAYOUT_A = QtWidgets.QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.setCentralWidget(self.FRAME_A)

        # Quit button
        self.quit_button = QtWidgets.QPushButton(text='Quit')
        setCustomSize(self.quit_button, 100, 50)
        self.quit_button.clicked.connect(self.quit_button_action)
        self.LAYOUT_A.addWidget(self.quit_button, *(6, 0, 1, 1))

        # record/record finish button
        self.record_button = QtWidgets.QPushButton(text='Start Record')
        setCustomSize(self.record_button, 100, 50)
        self.record_button.clicked.connect(self.record_button_action)
        self.LAYOUT_A.addWidget(self.record_button, *(6, 6, 1, 1))

        # report button
        self.report_button = QtWidgets.QPushButton(text='Report Analysis')
        setCustomSize(self.report_button, 100, 50)
        self.report_button.clicked.connect(self.report_button_action)
        self.LAYOUT_A.addWidget(self.report_button, *(6, 11, 1, 1))

        # subject name input
        self.subject_name = QtWidgets.QLineEdit()
        setCustomSize(self.subject_name, 200, 50)
        # self.report_button.clicked.connect(self.report_button_action)
        self.LAYOUT_A.addWidget(self.subject_name, *(6, 4, 1, 2))

        # Add Picture emotion
        self.emotion_picture = QtWidgets.QLabel()
        # self.emotion_picture.setStyleSheet("border: 1px solid black")
        self.emotion_picture.setAlignment(QtCore.Qt.AlignCenter)
        setCustomSize(self.emotion_picture, 4 * umulti, 3 * umulti)
        self.LAYOUT_A.addWidget(self.emotion_picture, *(0, 0, 3, 4))

        # Connection Status
        self.text_display = QtWidgets.QLabel()
        self.text_display.setStyleSheet("border: 1px solid black")
        setCustomSize(self.text_display, 4 * umulti, 3 * umulti)
        self.LAYOUT_A.addWidget(self.text_display, *(3, 0, 3, 4))

        # Place the matplotlib figure
        # self.myFig = CustomFigCanvas('Mean All', y_scale=[3800, 4800])
        # self.myFig = CustomFigCanvas('Arousal/Valence', y_scale=[0, 4])
        self.myFig = CustomFigCanvas2('Arousal/Valence', y_scale=[0, 4])
        # self.myFig = CustomFigCanvas2('Arousal/Valence-EEG', y_scale=[0, 15])
        self.LAYOUT_A.addWidget(self.myFig, *(0, 4, 3, 8)) # span row 1 column 2

        # self.myFig2 = CustomFigCanvas('Mean Best 6', y_scale=[3800, 4800])
        self.myFig2 = CustomFigCanvas('Mean EEG', y_scale=[0, 1000])
        self.LAYOUT_A.addWidget(self.myFig2, *(3, 4, 3, 8)) # span row 1 column 2

        self.record = False

        self.record_info = None

        # Add the callbackfunc to ..
        # myDataLoop = threading.Thread(name='myDataLoop', target=run_process, daemon=True, args=(self.addData_callbackFunc,))
        # myDataLoop.start()
        # im = Image.open('Penguins.jpg')
        # im = im.convert("RGBA")
        # data = im.tobytes("raw", "RGBA")
        # qim = QtGui.QImage(data, im.size[0], im.size[1], QtGui.QImage.Format_ARGB32)
        # pix = QtGui.QPixmap.fromImage(qim)

        self.show()

    ''''''


    def zoomBtnAction(self):
        print("zoom in")
        self.myFig.zoomIn(5)

    def quit_button_action(self):
        sys.exit()

    def record_button_action(self):
        # print('record')
        if len(self.subject_name.text()) == 0:
            self.show_dialog()
        else:
            self.record = not self.record

        if self.record:
            self.record_button.setText('Stop Recording')
        else:
            self.record_button.setText('Start Record')
            self.show_report_dialog()

    def report_button_action(self):
        self.show_report_dialog()

    def show_dialog(self):
        self.d = QtWidgets.QDialog()
        label = QtWidgets.QLabel('Please enter a name...')
        b1 = QtWidgets.QPushButton(text='OK')
        b1.move(50, 50)
        b1.clicked.connect(self.close_dialog)
        layout = QtWidgets.QGridLayout()
        layout.addWidget(label, 0, 0)
        layout.addWidget(b1, 1, 0)
        self.d.setGeometry(600, 300, 200, 100)
        self.d.setLayout(layout)
        self.d.setWindowTitle('Warning:')
        self.d.setWindowModality(QtCore.Qt.ApplicationModal)
        self.d.exec_()

    def show_report_dialog(self):
        report_played_time, report_fun, report_immersion, report_difficulty, report_emotion,\
            overall_estimation = self.make_final_analysis_text()

        self.d = QtWidgets.QDialog()
        label_header = QtWidgets.QLabel('Game Play EEG Anlaysis')

        label_fun = QtWidgets.QLabel(report_fun)
        label_fun.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        label_immersion = QtWidgets.QLabel(report_immersion)
        label_immersion.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        label_difficulty = QtWidgets.QLabel(report_difficulty)
        label_difficulty.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        label_emotion = QtWidgets.QLabel(report_emotion)
        label_emotion.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignLeft)
        label_overall_estimation = QtWidgets.QLabel(overall_estimation)
        label_overall_estimation.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        b1 = QtWidgets.QPushButton(text='Confirm')
        b1.move(50, 50)
        b1.clicked.connect(self.close_dialog)

        layout = QtWidgets.QGridLayout()
        layout.addWidget(label_header, *(0, 0, 1, 8))
        layout.addWidget(label_fun, *(1, 0, 1, 2))
        layout.addWidget(label_immersion, *(1, 2, 1, 2))
        layout.addWidget(label_difficulty, *(1, 4, 1, 2))
        layout.addWidget(label_emotion, *(1, 6, 1, 2))
        layout.addWidget(label_overall_estimation, *(2, 0, 3, 8))
        layout.addWidget(b1, 5, 3, 1, 2)
        self.d.setGeometry(600, 300, 800, 600)
        self.d.setLayout(layout)
        self.d.setWindowTitle('Report Analysis')
        self.d.setWindowModality(QtCore.Qt.ApplicationModal)
        self.d.exec_()

    def close_dialog(self):
        self.d.close()


    ''''''

    def addData_callbackFunc(self, value):
        # print("Add data: " + str(value))
        # print(value)
        self.record_info = value

        final_emotion, eeg_realtime, connection_status, disconnected_list, arousal_all, valence_all, \
        fun_stat, immersion_stat, difficulty_stat, emotion_stat, fun_stat_record, immersion_stat_record, \
        difficulty_stat_record, emotion_stat_record, fun_records, immersion_records, difficulty_records, \
        emotion_records, record_start_time = self.retrieve_info(self.record_info)

        if len(disconnected_list) == 0:
            disconnected_list = ['None']

        if self.get_record_status():
            record_status = 'ON'
            text_display_analysis = '\n\n[Fun] ' + self.make_analysis_text(fun_stat_record) \
                                    + '\n\n[Immersion] ' + self.make_analysis_text(immersion_stat_record) \
                                    + '\n\n[Difficulty] ' + self.make_analysis_text(difficulty_stat_record) \
                                    + '\n\n[Emotion] ' + self.make_analysis_text(emotion_stat_record)

        else:
            record_status = 'OFF'
            text_display_analysis = '\n\n[Fun] (Dominant over last 5 seconds) \n' + fun_stat \
                                    + '\n\n[Immersion] (Dominant over last 5 seconds) \n' + immersion_stat \
                                    + '\n\n[Difficulty] (Dominant over last 5 seconds) \n' + difficulty_stat \
                                    + '\n\n[Emotion] (Dominant over last 5 seconds) \n' + emotion_stat


        # new_eeg = (np.mean(eeg_realtime, axis=0) - np.min(eeg_realtime, axis=0)) / 100
        # new_eeg_realtime1 = new_eeg * np.mean(arousal_all, axis=0)
        # new_eeg_realtime2 = new_eeg * np.mean(valence_all, axis=0)
        arousal_realtime = np.mean(arousal_all, axis=0)
        valence_realtime = np.mean(valence_all, axis=0)
        new_eeg_realtime = np.mean(eeg_realtime, axis=0) - 3800
        text_display_detail = '[Disconnected Electrodes]' + '(Signal Quality: ' + str(int(connection_status)) + '%)\n' \
                              + ' '.join(disconnected_list) \
                              + '\n\n[Record/Analysis Status] ' + record_status \
                              + text_display_analysis

        self.myFig.addData(arousal_realtime, valence_realtime)
        # self.myFig.addData(new_eeg_realtime1)
        self.myFig2.addData(new_eeg_realtime)
        # self.emotion_label.setText(str(final_emotion))
        pixmap = QtGui.QPixmap(join(image_path, str(int(final_emotion)) + '.png'))
        pixmap = pixmap.scaledToWidth(200)
        pixmap = pixmap.scaledToHeight(200)
        # pixmap.scaled(200, 200, QtCore.Qt.KeepAspectRatio)
        self.emotion_picture.setPixmap(pixmap)
        self.text_display.setText(text_display_detail)
        # self.text_display.setText(str(float(connection_status)) + '\ndisconnected list:')

    def retrieve_info(self, value):
        final_emotion = value['final_emotion']
        eeg_realtime = value['eeg_realtime']
        connection_status = value['connection_status']
        disconnected_list = value['disconnected_list']
        arousal_all = value['arousal_all']
        valence_all = value['valence_all']
        fun_stat = value['fun_stat']
        immersion_stat = value['immersion_stat']
        difficulty_stat = value['difficulty_stat']
        emotion_stat = value['emotion_stat']
        fun_stat_record = value['fun_stat_record']
        immersion_stat_record = value['immersion_stat_record']
        difficulty_stat_record = value['difficulty_stat_record']
        emotion_stat_record = value['emotion_stat_record']
        fun_records = value['fun_records']
        immersion_records = value['immersion_records']
        difficulty_records = value['difficulty_records']
        emotion_records = value['emotion_records']
        record_start_time = value['record_start_time']

        return final_emotion, eeg_realtime, connection_status, disconnected_list, arousal_all, valence_all,\
               fun_stat, immersion_stat, difficulty_stat, emotion_stat, fun_stat_record, immersion_stat_record,\
               difficulty_stat_record, emotion_stat_record, fun_records, immersion_records, difficulty_records,\
               emotion_records, record_start_time

    def make_analysis_text(self, data, ratio_stat=False, duration=0):
        """
        Make analysis details with percentage information and its duration for each component

        :param data:
        :return:
        """
        txt = ''

        if ratio_stat and duration != 0:
            total_count = 0
            for k, v in data.items():
                total_count += v

            for k, v in data.items():
                percentage = str(int(v * 100 / total_count))
                final_duration = str(int(duration * v / total_count))
                if len(txt) == 0:
                    txt = txt + str(k) + ' (' + percentage + '%, ' + final_duration + 's)'
                else:
                    txt = txt + ' ' + str(k) + ' (' + percentage + '%, ' + final_duration + 's)'
        else:
            for k, v in data.items():
                txt = txt + '\n' + str(k) + ': ' + str(v)

        return txt


    def make_final_analysis_text(self):
        final_emotion, eeg_realtime, connection_status, disconnected_list, arousal_all, valence_all, \
        fun_stat, immersion_stat, difficulty_stat, emotion_stat, fun_stat_record, immersion_stat_record, \
        difficulty_stat_record, emotion_stat_record, fun_records, immersion_records, difficulty_records, \
        emotion_records, record_start_time = self.retrieve_info(self.record_info)

        total_played_time = (datetime.now() - record_start_time).seconds

        report_played_time = 'Total Played Time: %is' % (total_played_time)
        report_fun = '[Fun]' + self.make_analysis_text(fun_stat_record)
        report_immersion = '[Immersion]' + self.make_analysis_text(immersion_stat_record)
        report_difficulty = '[Difficulty]' + self.make_analysis_text(difficulty_stat_record)
        report_emotion = '[Emotion]' + self.make_analysis_text(emotion_stat_record)

        overall_estimation = '[Overall Estimation]\n' + 'Total Played Time: ' + str(total_played_time) + ' seconds' \
                             + '\nFun Moments:' + self.make_analysis_text(fun_stat_record, ratio_stat=True,
                                                                          duration=total_played_time) \
                             + '\nImmersion Moments:' + self.make_analysis_text(immersion_stat_record, ratio_stat=True,
                                                                                duration=total_played_time) \
                             + '\nDifficulty Moments:' + self.make_analysis_text(difficulty_stat_record, ratio_stat=True,
                                                                                 duration=total_played_time) \
                             + '\nEmotion Moments:' + self.make_analysis_text(emotion_stat_record, ratio_stat=True,
                                                                              duration=total_played_time)

        return report_played_time, report_fun, report_immersion, report_difficulty, report_emotion, overall_estimation

    def get_record_status(self):
        return self.record

    def get_subject_name(self):
        return self.subject_name.text()




''' End Class '''


class CustomFigCanvas(FigureCanvas, TimedAnimation):

    def __init__(self, title, y_scale):

        self.addedData = []
        print(matplotlib.__version__)

        # The data
        self.xlim = 200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50

        # The window
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_title(title)


        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='red', linewidth=2)
        self.line1_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(y_scale[0], y_scale[1])

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=50, blit = True)

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        for l in lines:
            l.set_data([], [])

    def addData(self, value):
        self.addedData.append(value)

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom,top)
        self.draw()


    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])

        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]


class CustomFigCanvas2(FigureCanvas, TimedAnimation):

    def __init__(self, title, y_scale):

        self.addedData = []
        self.addedData2 = []
        print(matplotlib.__version__)

        # The data
        self.xlim = 200
        self.n = np.linspace(0, self.xlim - 1, self.xlim)
        a = []
        b = []
        a.append(2.0)
        a.append(4.0)
        a.append(2.0)
        b.append(4.0)
        b.append(3.0)
        b.append(4.0)
        self.y = (self.n * 0.0) + 50
        self.y2 = (self.n * 0.0) + 50

        # The window
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_title(title)


        # self.ax1 settings
        self.ax1.set_xlabel('time')
        self.ax1.set_ylabel('raw data')
        self.line1 = Line2D([], [], color='blue')
        self.line1_tail = Line2D([], [], color='blue', linewidth=2)
        self.line1_head = Line2D([], [], color='blue', marker='o', markeredgecolor='r')
        self.line2 = Line2D([], [], color='red')
        self.line2_tail = Line2D([], [], color='red', linewidth=2)
        self.line2_head = Line2D([], [], color='red', marker='o', markeredgecolor='r')
        self.ax1.add_line(self.line1)
        self.ax1.add_line(self.line1_tail)
        self.ax1.add_line(self.line1_head)
        self.ax1.add_line(self.line2)
        self.ax1.add_line(self.line2_tail)
        self.ax1.add_line(self.line2_head)
        self.ax1.set_xlim(0, self.xlim - 1)
        self.ax1.set_ylim(y_scale[0], y_scale[1])

        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval=50, blit = True)

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        lines = [self.line1, self.line1_tail, self.line1_head]
        lines2 = [self.line2, self.line2_tail, self.line2_head]
        for l, l2 in zip(lines, lines2):
            l.set_data([], [])
            l2.set_data([], [])

    def addData(self, value, value2):
        self.addedData.append(value)
        self.addedData2.append(value2)

    def zoomIn(self, value):
        bottom = self.ax1.get_ylim()[0]
        top = self.ax1.get_ylim()[1]
        bottom += value
        top -= value
        self.ax1.set_ylim(bottom,top)
        self.draw()

    def _step(self, *args):
        # Extends the _step() method for the TimedAnimation class.
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            self.abc += 1
            print(str(self.abc))
            TimedAnimation._stop(self)
            pass

    def _draw_frame(self, framedata):
        margin = 2
        while(len(self.addedData) > 0):
            self.y = np.roll(self.y, -1)
            self.y[-1] = self.addedData[0]
            del(self.addedData[0])

        self.line1.set_data(self.n[ 0 : self.n.size - margin ], self.y[ 0 : self.n.size - margin ])
        self.line1_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]), np.append(self.y[-10:-1 - margin], self.y[-1 - margin]))
        self.line1_head.set_data(self.n[-1 - margin], self.y[-1 - margin])
        # self._drawn_artists = [self.line1, self.line1_tail, self.line1_head]

        while (len(self.addedData2) > 0):
            self.y2 = np.roll(self.y2, -1)
            self.y2[-1] = self.addedData2[0]
            del (self.addedData2[0])

        self.line2.set_data(self.n[0: self.n.size - margin], self.y2[0: self.n.size - margin])
        self.line2_tail.set_data(np.append(self.n[-10:-1 - margin], self.n[-1 - margin]),
                                 np.append(self.y2[-10:-1 - margin], self.y2[-1 - margin]))
        self.line2_head.set_data(self.n[-1 - margin], self.y2[-1 - margin])
        # self._drawn_artists = [self.line1, self.line1_tail, self.line1_head, self.line2, self.line2_tail, self.line2_head]


''' End Class '''


# You need to setup a signal slot mechanism, to
# send data to your GUI in a thread-safe way.
# Believe me, if you don't do this right, things
# go very very wrong..
class Communicate(QtCore.QObject):
    # data_signal = QtCore.pyqtSignal(float)
    # data_signal = QtCore.pyqtSignal(np.ndarray)
    data_signal = QtCore.pyqtSignal(object)

''' End Class '''



def dataSendLoop(addData_callbackFunc, stat):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    # Simulate some data
    n = np.linspace(0, 499, 500)
    y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))
    i = 0

    while(True):
        if(i > 499):
            i = 0
        time.sleep(0.001)
        # mySrc.data_signal.emit(y[i]) # <- Here you emit a signal!
        i += 1
    ###
###


def draw_graph(run_process=None):
    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create('Plastique'))
    # myGUI = CustomMainWindow(run_process)
    myGUI = CustomMainWindow()
    myDataLoop = threading.Thread(name='myDataLoop', target=run_process, daemon=True,
                                  args=(myGUI.addData_callbackFunc, myGUI.get_record_status, myGUI.get_subject_name))
    myDataLoop.start()

    sys.exit(app.exec_())


if __name__== '__main__':
    draw_graph()

''''''