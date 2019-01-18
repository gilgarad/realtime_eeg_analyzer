import json
import os

'''
CONST
'''
ROOT_PATH = os.path.dirname(os.path.realpath(__file__))
config = json.load(open(os.path.join(ROOT_PATH, 'config', 'system_config.json')))

# DB_ADDR = config['db-addr']
# FILE_ROOT_DIR = config['file-root-dir']
# ALLOWED_FILTYPE_CATEGORY = ['rawdata', 'book', 'cohesion', 'dictionary', 'postag', 'wordvectorize']
LOG_LEVEL = config['log-level']

HOST = config['host']
PORT = int(config['port'])
# ROOT_DIR = config['root-directory']
# MODEL_CONFIG = config['model-config']

