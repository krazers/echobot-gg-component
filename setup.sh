#!/bin/bash
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root / with sudo!" 
   exit 1
fi

echo "############### Install Jetbot libraries ###############" 

#sudo apt-get install python3-pip -y
#sudo pip3 install setuptools
#git clone https://github.com/NVIDIA-AI-IOT/jetbot
#./jetcard/install.sh
#sudo python3 jetbot/setup.py install

echo "############### FULLY COMPLETE - CHECK FOR ERRORS ###############" 
exit
