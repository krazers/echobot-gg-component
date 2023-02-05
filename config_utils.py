from logging import INFO, StreamHandler, getLogger
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
TIMEOUT = 10
SCORE_CONVERTER = 255

#Topics
EchoBotStatusUpdateSubscribe = ""
EchoBotStatusGetSubscribe = ""
EchoBotDetectionsPublish = ""
EchoBotStatusUpdatePublish = ""
EchoBotStatusGetPublish = ""

# Get a logger
logger = getLogger()
handler = StreamHandler(stdout)
logger.setLevel(INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(handler)