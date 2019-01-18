# from utils.database import DatabaseController
# from const import DB_ADDR, LOG_LEVEL, ROOT_PATH, DNLPY_PATH
from const import LOG_LEVEL, ROOT_PATH
import os
import logging
import sys
# from utils.mongo_logger import MongoLoggerHandler

# sys.path.append(DNLPY_PATH)

log_switch = {
    'debug'     : logging.DEBUG,
    'warning'   : logging.WARNING,
    'warn'      : logging.WARN,
    'info'      : logging.INFO,
    'error'     : logging.ERROR,
    'critical'  : logging.CRITICAL
}

'''
Initial Declaration
'''
# dbc = DatabaseController(DB_ADDR)

logger = logging.getLogger('realtime_emotion')

formatter = logging.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

fileHandler = logging.FileHandler(os.path.join(ROOT_PATH, 'realtime_emotion.log'))
streamHandler = logging.StreamHandler()

fileHandler.setFormatter(formatter)
streamHandler.setFormatter(formatter)

logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
# logger.addHandler(MongoLoggerHandler(dbc=dbc))

logger.setLevel(log_switch[LOG_LEVEL])



