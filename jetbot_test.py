#!/usr/bin/python3.6
import config_utils
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

config_utils.logger.info("Libraries loaded")

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
        
    # otherwsie steer towards target
    else:
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        robot.set_motors(
            float(speed + turn_gain * center[0]),
            float(speed - turn_gain * center[0])
        )
        
 
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
    else:
        robot.left(0.4)
    
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
    config_utils.logger.info("Starting avoid obstacles routine")
    camera.unobserve_all()
    update({'new': camera.value})
    camera.observe(update, names='value')  
 
def start_object_following():
    config_utils.logger.info("Starting object follow routine")
    camera.unobserve_all()
    camera.observe(execute, names='value')
    
def stop_object_following():
    global robot, camera
    config_utils.logger.info("Stopping object following routing")
    camera.unobserve_all()
    time.sleep(1.0)
    robot.stop()
    config_utils.logger.info("EchoBot stopped")
      
config_utils.logger.info("Startup, updating mode and speed shadow")
mode = "stop"

config_utils.logger.info("Loading object detector. This will take a minute...")
from jetbot import ObjectDetector

######## Later this can be downloaded form another component
##############################################################
config_utils.logger.info("Loading coco model...")

model = ObjectDetector('/echobot/ssd_mobilenet_v2_coco.engine')
from jetbot import bgr8_to_jpeg
config_utils.logger.info("Loading camera")
from jetbot import Camera
camera = Camera.instance(width=300, height=300)
from jetbot import Robot

# Initialize robot
config_utils.logger.info("Initialize robot library")
robot = Robot()

config_utils.logger.info("Loading model")
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

start_avoid_obstacles()

while True:
    time.sleep(5)