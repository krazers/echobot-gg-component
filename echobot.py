#!/usr/bin/python3.6
from logging import INFO, DEBUG, StreamHandler, getLogger
import logging
from json import dumps, loads
from os import getenv
import sys
import time
import json
from socket import socket, AF_INET, SOCK_DGRAM, IPPROTO_UDP, timeout
import socket
import numpy as np
import cv2
import torch
import torchvision
import torch.nn.functional as F
import datetime
import os
import threading
import awsiot.greengrasscoreipc.client as client
from awsiot.eventstreamrpc import Connection, LifecycleHandler, MessageAmendment
from awsiot.greengrasscoreipc.model import (
    IoTCoreMessage,
    QOS,
    ConfigurationUpdateEvents,
    GetConfigurationRequest,
    PublishToIoTCoreRequest,
    PublishToTopicRequest,
    PublishMessage,
    JsonMessage,
    GetThingShadowRequest,
    UpdateThingShadowRequest,
    SubscribeToTopicRequest,
    SubscriptionResponseMessage,
    SubscribeToIoTCoreRequest
)
from awscrt.io import (
    ClientBootstrap,
    DefaultHostResolver,
    EventLoopGroup,
    SocketDomain,
    SocketOptions,
)

# Get a logger
logger = getLogger()
logging.basicConfig(stream=sys.stdout, level=INFO)
logger.info("Libraries loaded")

# Global variables
default_speed = 0.3 
speed = 0 #Current Speed
turn_gain = 0.8
width = int(300)
height = int(300)
filespath = "/echobot/"
null = None

# Timeout and QOS for pubilshing/subscription events with IPC
TIMEOUT = 15
QOS_TYPE = QOS.AT_LEAST_ONCE

class IPCUtils:
    def connect(self):
        elg = EventLoopGroup()
        resolver = DefaultHostResolver(elg)
        bootstrap = ClientBootstrap(elg, resolver)
        socket_options = SocketOptions()
        socket_options.domain = SocketDomain.Local
        amender = MessageAmendment.create_static_authtoken_amender(getenv("SVCUID"))
        hostname = getenv("AWS_GG_NUCLEUS_DOMAIN_SOCKET_FILEPATH_FOR_COMPONENT")
        connection = Connection(
            host_name=hostname,
            port=8033,
            bootstrap=bootstrap,
            socket_options=socket_options,
            connect_message_amender=amender,
        )
        self.lifecycle_handler = LifecycleHandler()
        connect_future = connection.connect(self.lifecycle_handler)
        connect_future.result(TIMEOUT)
        return connection

    def publish_results_to_cloud(self, topic, PAYLOAD):
        r"""
        Ipc client creates a request and activates the operation to publish messages to the IoT core
        with a qos type over a topic.
        :param PAYLOAD: An dictionary object with inference results.
        """
        try:
            request = PublishToIoTCoreRequest(
                topic_name=topic,
                qos=QOS_TYPE,
                payload=dumps(PAYLOAD).encode(),
            )
            operation = ipc_client.new_publish_to_iot_core()
            operation.activate(request).result(TIMEOUT)
            logger.info("Publishing results to the IoT core...")
            operation.get_response().result(TIMEOUT)
        except Exception as e:
            logger.error(str(e))
            logger.error("Exception occured during publish: {}".format(str(e)))

    def subscribe_to_cloud(self, topic, messagetype):
        r"""
        Subcription to topics in IoT Core directly. 
        Note: Passing message type as a way to detect with handler to use. 
        Passing handler as a argument didn't appear to work.
        """
        try:
            logger.info("Subscribing to Topic: {}".format(topic))
            request = SubscribeToIoTCoreRequest()
            request.topic_name = topic
            request.qos = QOS_TYPE
            handler = UpdatedShadowStreamHandler()
            if(messagetype == 2):
                handler = GetShadowStreamHandler()            
            operation = ipc_client.new_subscribe_to_iot_core(handler) 
            future = operation.activate(request)
            future.result(TIMEOUT)
            logger.info("Subscribed to Topic: {}".format(topic))
        except Exception as e:
            logger.error(
                "Exception occured during subscription: {}".format(str(e))
            )
            exit(1)

    def get_configuration(self):
        r"""
        Ipc client creates a request and activates the operation to get the configuration of
        inference component passed in its recipe.
        :return: A dictionary object of DefaultConfiguration from the recipe.
        """
        try:
            request = GetConfigurationRequest()
            operation = ipc_client.new_get_configuration()
            operation.activate(request).result(TIMEOUT)
            result = operation.get_response().result(TIMEOUT)
            return result.value
        except Exception as e:
            logger.error(
                "Exception occured during fetching the configuration: {}".format(str(e))
            )
            exit(1)
    

class GetShadowStreamHandler(client.SubscribeToIoTCoreStreamHandler):
    def __init__(self):
        super().__init__()
    
    def on_stream_event(self, event: IoTCoreMessage) -> None:
        r"""
        On messages received aft, look at delta changes to detect
        which mode to be on
        """
        global mode
        logger.info(event.message.payload)
        message = json.loads(event.message.payload)
        if("desired" in message["state"]):
            if("mode" in message["state"]["desired"]):
                if(mode != message["state"]["desired"]["mode"]):
                    mode = message["state"]["desired"]["mode"]
                    logger.info("Current mode is {}".format(mode))
                    if(mode == "stop"):
                        stop_object_following()
                    elif(mode == "follow"):
                        threading.Thread(target=start_object_following, args=()).start()
                    elif(mode == "avoidobstacles"):
                        threading.Thread(target=start_avoid_obstacles, args=()).start()

    def on_stream_error(self, error: Exception) -> bool:
        # Handle error.
        logger.error("Exception on GetShadow Stream Handler: {}".format(error))
        return True  # Return True to close stream, False to keep stream open.

    def on_stream_closed(self) -> None:
        # Handle close.
        pass


class UpdatedShadowStreamHandler(client.SubscribeToIoTCoreStreamHandler):
    def __init__(self):
        super().__init__()

    def on_stream_event(self, event: IoTCoreMessage) -> None:
        global mode    
        logger.info(event.message.payload)
        message = json.loads(event.message.payload)
        if("mode" in message["state"]):
            if(mode != message["state"]["mode"]):
                mode = message["state"]["mode"]
                logger.info("Mode changed to {}".format(mode))
                if(mode == "stop"):
                    threading.Thread(target=stop_object_following, args=()).start()
                elif(mode == "follow"):
                    threading.Thread(target=start_object_following, args=()).start()
                elif(mode == "avoidobstacles"):
                    threading.Thread(target=start_avoid_obstacles, args=()).start()
        if("speed" in message["state"]):
            update_speed(float(message["state"]["speed"]))
        if("command" in message["state"]):
            update_command(message["state"]["command"])

    def on_stream_error(self, error: Exception) -> bool:
        logger.error("Exception on UpdateShadow Stream Handler: {}".format(error))
        return True  # Return True to close stream, False to keep stream open.

    def on_stream_closed(self) -> None:
        # Handle close.
        pass



def report_detections(blocked, detectioncount, following):
    message = {
        "timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f+00:00'),
        "blocked": blocked,
        "detected_object_count": detectioncount,
        "following": following,
        "speed": speed,
        "mode": mode
    }
    IPCUtils().publish_results_to_cloud(config["EchoBotDetectionsPublish"], message)

def update_mode(currentmode):
    # Update shadow with current state
    message = {
        "state": {
            "desired":{
                "mode": null
            },
            "reported":{
                "mode":currentmode,
                "speed":speed
            }
        }
    }
    IPCUtils().publish_results_to_cloud(config["EchoBotStatusUpdatePublish"], message)


def update_speed(currentspeed):
    global speed
    # Update shadow with current state
    speed = currentspeed
    message = {
        "state": {
            "desired":{
                "speed": null
            },
            "reported":{
                "mode":mode,
                "speed":currentspeed
            }
        }
    }
    IPCUtils().publish_results_to_cloud(config["EchoBotStatusUpdatePublish"], message)
   

def update_command(currentcommand):
    # Update shadow with current state
    turn_command(currentcommand)
    message = {
        "state": {
            "desired":{
                "command": null
            }
        }
    }
    IPCUtils().publish_results_to_cloud(config["EchoBotStatusUpdatePublish"], message)


def update_status(status):
    # Update shadow with current state
    testIP = "8.8.8.8"
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect((testIP, 0))
    ipaddr = s.getsockname()[0]

    message = {
        "state": {
            "reported":{
                "status":status,
                "ip": ipaddr
            }
        }
    }
    IPCUtils().publish_results_to_cloud(config["EchoBotStatusUpdatePublish"], message)

  
def preprocess1(camera_value, device, normalize):
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x    
    
def preprocess2(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x
    
def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)
    
def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0]**2 + vec[1]**2)

def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection
    
def execute(change):

    image = change['new']
    
    # execute collision model to determine if blocked
    collision_output = collision_model(preprocess1(image, device, normalize)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
    
    # turn left if blocked
    if prob_blocked > 0.5:
        robot.left(0.3)
        #report_detections(True,1,False)
        #image_widget.value = bgr8_to_jpeg(image)
        return
        
    # compute all detected objects
    detections = model(image)
    
    # draw all detections on image
    #for det in detections[0]:
    #    bbox = det['bbox']
    #    cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
    
    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] == int(1)]
    
    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    #if det is not None:
    #    bbox = det['bbox']
    #    cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)
    
    
    # otherwise go forward if no target detected
    if det is None:
        robot.forward(float(speed))
        #report_detections(False,0,False)
        
    # otherwsie steer towards target
    else:
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        robot.set_motors(
            float(speed + turn_gain * center[0]),
            float(speed - turn_gain * center[0])
        )
        #report_detections(False,len(detections),True)
        
 
# Avoid obstacles
def update(change):
    x = change['new'] 
    
    x = preprocess2(x)
    y = collision_model(x)
    
    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    
    prob_blocked = float(y.flatten()[0])
    
    if prob_blocked < 0.5:
        robot.forward(float(speed))
        #report_detections(False,1,False)
    else:
        robot.left(0.4)
        #report_detections(True,1,False)
    
    time.sleep(0.001)       

def turn_command(command):
    try:
        if(command != ""):
            if(command == "left"):
                robot.left(0.4)
                time.sleep(0.5)
            elif(command == "right"):
                robot.right(0.4)
                time.sleep(0.5)
            elif(command == "around"):
                robot.left(0.4)
                time.sleep(1)
            else:
                robot.stop()
    except:
        pass
    
def start_avoid_obstacles():
    logger.info("Starting avoid obstacles routine")
    update_speed(default_speed)
    update_mode("avoidobstacles")
    camera.unobserve_all()
    update({'new': camera.value})
    camera.observe(update, names='value')  
 
def start_object_following():
    logger.info("Starting object follow routine")
    update_speed(default_speed)
    update_mode("follow")
    camera.unobserve_all()
    camera.observe(execute, names='value')
    
def stop_object_following():
    global robot, camera
    logger.info("Stopping object following routing")
    update_speed(0)
    update_mode("stop")
    camera.unobserve_all()
    time.sleep(1.0)
    robot.stop()
    logger.info("EchoBot stopped")


##############################################################
## Startup - Notify that object detector is loading
##############################################################
# Get the ipc client
try:
    ipc_client = client.GreengrassCoreIPCClient(IPCUtils().connect())
    logger.info("Created IPC client...")
except Exception as e:
    logger.error(
        "Exception occured during the creation of an IPC client: {}".format(str(e))
    )
    exit(1)
  
logger.info("Loading recipe parameters...")
config = IPCUtils().get_configuration() 
#print(config["EchoBotStatusUpdatePublish"])

logger.info("Startup, updating mode and speed shadow")
mode = "stop"
update_mode("stop")
update_speed(speed)
logger.info("Loading object detector. This will take a minute...")
update_status("Loading object detector")
from jetbot import ObjectDetector

##############################################################
## Initialize Robot and Model
## Later model can be downloaded form another GG component
##############################################################
logger.info("Loading coco model...")

# Load object detector for following
model = ObjectDetector('/echobot/ssd_mobilenet_v2_coco.engine')

from jetbot import bgr8_to_jpeg
logger.info("Loading camera")
update_status("Loading camera")
from jetbot import Camera
camera = Camera.instance(width=300, height=300)
from jetbot import Robot

# Initialize robot
logger.info("Initialize robot library")
robot = Robot()

# Load Model for collision avoidance
logger.info("Loading model")
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)

collision_model.load_state_dict(torch.load('/echobot/best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

##############################################################
## Subscribe to topics to receive commands
##############################################################
IPCUtils().subscribe_to_cloud(config["EchoBotStatusUpdateSubscribe"], 1)
time.sleep(2)
IPCUtils().subscribe_to_cloud(config["EchoBotStatusGetSubscribe"], 2)
time.sleep(2)
update_status("Ready for commands")

##############################################################
## Publish message to get current shadow state
##############################################################
IPCUtils().publish_results_to_cloud(config["EchoBotStatusGetPublish"], "")

while True:
    time.sleep(5)