#!/usr/bin/python3.6
import IPCUtils as ipc_utils
from http_parser.parser import HttpParser
import argparse
import requests 
import time
import json
from pathlib import Path
from socket import socket, AF_INET, SOCK_DGRAM, IPPROTO_UDP, timeout
import subprocess
import threading
import fileinput
import numpy as np
import cv2
import math
import colorconverter
import config_utils
import torch
import torchvision
import torch.nn.functional as F
import datetime
import os
import sys
import uuid
import logging

# Global variables
speed = 0.4 #normal
turn_gain = 0.8
width = int(300)
height = int(300)
MAX_DISCOVERY_RETRIES = 10   
retryCount = MAX_DISCOVERY_RETRIES
discovered = False
groupCA = None
coreInfo = None
filespath = "/echobot/"
null = None

# General message notification callback
def GetShadowResponse(client, userdata, message):
    global mode
    if("desired" in json.loads(message.payload)["state"]):
        if("mode" in json.loads(message.payload)["state"]["desired"]):
            if(mode != json.loads(message.payload)["state"]["desired"]["mode"]):
                mode = json.loads(message.payload)["state"]["desired"]["mode"]
                print("Current mode is {}".format(mode))
                if(mode == "stop"):
                    stop_object_following()
                elif(mode == "follow"):
                    start_object_following()
                elif(mode == "avoidobstacles"):
                    start_avoid_obstacles()
    
#Delta change
def UpdatedShadowResponse(client, userdata, message):
    global mode    
    if("mode" in json.loads(message.payload)["state"]):
        if(mode != json.loads(message.payload)["state"]["mode"]):
            mode = json.loads(message.payload)["state"]["mode"]
            print("Mode changed to {}".format(mode))
            if(mode == "stop"):
                stop_object_following()
            elif(mode == "follow"):
                start_object_following()
            elif(mode == "avoidobstacles"):
                start_avoid_obstacles()
    if("speed" in json.loads(message.payload)["state"]):
        update_speed(float(json.loads(message.payload)["state"]["speed"]))
    if("command" in json.loads(message.payload)["state"]):
        update_command(json.loads(message.payload)["state"]["command"])

def report_detections(blocked, detectioncount, following):
    message = {
        "timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f+00:00'),
        "blocked": blocked,
        "detected_object_count": detectioncount,
        "following": following,
        "speed": speed,
        "mode": mode
    }
    ipc_utils.IPCUtils().publish_results_to_pubsub_ipc(config_utils.EchoBotDetectionsPublish, json.dumps(message))

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
    ipc_utils.IPCUtils().publish_results_to_pubsub_ipc(config_utils.EchoBotStatusUpdatePublish, json.dumps(message))


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
    ipc_utils.IPCUtils().publish_results_to_pubsub_ipc(config_utils.EchoBotStatusUpdatePublish, json.dumps(message))
   

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
    ipc_utils.IPCUtils().publish_results_to_pubsub_ipc(config_utils.EchoBotStatusUpdatePublish, json.dumps(message))


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
    ipc_utils.IPCUtils().publish_results_to_pubsub_ipc(config_utils.EchoBotStatusUpdatePublish, json.dumps(message))

  
    
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
        report_detections(True,1,False)
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
        report_detections(False,0,False)
        
    # otherwsie steer towards target
    else:
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        robot.set_motors(
            float(speed + turn_gain * center[0]),
            float(speed - turn_gain * center[0])
        )
        report_detections(False,len(detections),True)
        
 
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
        report_detections(False,1,False)
    else:
        robot.left(0.4)
        report_detections(True,1,False)
    
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
            robot.stop()
    except:
        pass
    
def start_avoid_obstacles():
    print("Starting avoid obstacles routine")
    update_mode("avoidobstacles")
    camera.unobserve_all()
    update({'new': camera.value})
    camera.observe(update, names='value')  
 
def start_object_following():
    print("Starting object follow routine")
    update_mode("follow")
    camera.unobserve_all()
    camera.observe(execute, names='value')
    
def stop_object_following():
    global robot, camera
    print("Stopping object following routing")
    update_mode("stop")
    camera.unobserve_all()
    time.sleep(1.0)
    robot.stop()
    print("EchoBot stopped")
    
def set_configuration(config):
    r"""
    Sets a new config object with the combination of updated and default configuration as applicable.
    Calls inference code with the new config and indicates that the configuration changed.
    """
    new_config = {}
    if "EchoBotStatusUpdateSubscribe" in config:
        config_utils.EchoBotStatusUpdateSubscribe = config["EchoBotStatusUpdateSubscribe"]
    elif "EchoBotStatusGetSubscribe" in config:
        config_utils.EchoBotStatusGetSubscribe = config["EchoBotStatusGetSubscribe"]
    elif "EchoBotDetectionsPublish" in config:
        config_utils.EchoBotDetectionsPublish = config["EchoBotDetectionsPublish"]
    elif "EchoBotStatusUpdatePublish" in config:
        config_utils.EchoBotStatusUpdatePublish = config["EchoBotStatusUpdatePublish"]
    elif "EchoBotStatusGetPublish" in config:
        config_utils.EchoBotStatusGetPublish = config["EchoBotStatusGetPublish"]
      
print("Loading recipe parameters...")
set_configuration(ipc_utils.IPCUtils().get_configuration())        

# Configure logging
print("Configuring Logger...")
logger = logging.getLogger("EchoBot")
logger.setLevel(logging.DEBUG)
streamHandler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
streamHandler.setFormatter(formatter)
logger.addHandler(streamHandler)


print("Startup, updating mode and speed shadow")
mode = "stop"
update_mode("stop")
update_speed(speed)
print("Loading object detector. This will take a minute...")
update_status("Loading object detector")
from jetbot import ObjectDetector

######## Later this can be downloaded form another component
##############################################################
print("Loading coco model...")
model = ObjectDetector('/echobot/ssd_mobilenet_v2_coco.engine')
from jetbot import bgr8_to_jpeg
print("Loading camera")
update_status("Loading camera")
from jetbot import Camera
camera = Camera.instance(width=300, height=300)
from jetbot import Robot

# Initialize robot
print("Initialize robot library")
robot = Robot()

print("Loading model")
collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)

######## Later this can be downloaded form another component
##############################################################
collision_model.load_state_dict(torch.load('/echobot/best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)
mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

# Subscribe to shadow topics
ipc_utils.IPCUtils().subscribe_to_cloud(config_utils.EchoBotStatusUpdateSubscribe)
time.sleep(2)
ipc_utils.IPCUtils().subscribe_to_cloud(config_utils.EchoBotStatusGetSubscribe)
time.sleep(2)
update_status("Ready for commands")

# Get current shadow
ipc_utils.IPCUtils().publish_results_to_pubsub_ipc(config_utils.EchoBotStatusGetPublish, "")

while True:
    time.sleep(5)