from logging import INFO, StreamHandler, getLogger
import logging
from os import environ, path
from sys import stdout

from awsiot.greengrasscoreipc.model import QOS

# Set all the constants
SCORE_THRESHOLD = 0.3
MAX_NO_OF_RESULTS = 5
HEIGHT = 512
WIDTH = 512
SHAPE = (HEIGHT, WIDTH)
QOS_TYPE = QOS.AT_LEAST_ONCE
TIMEOUT = 15
SCORE_CONVERTER = 255

# Get a logger
logger = getLogger()
logger.setLevel(DEBUG)