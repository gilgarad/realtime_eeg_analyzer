""" subject.py: DAO object that contains the subject(user) information """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"


class Subject:
    def __init__(self):
        """ Initialize Subject object

        """
        self.name = ''
        self.age = 0