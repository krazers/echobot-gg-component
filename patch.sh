#!/bin/bash
if [[ $EUID -ne 0 ]]; then
   echo "This script must be run as root / with sudo!" 
   exit 1
fi

echo "############### Fixing Edimax WiFi Driver ###############" 
echo "blacklist rtl8192cu" | sudo tee -a /etc/modprobe.d/blacklist.conf
#Installing Patch https://github.com/pvaret/rtl8192cu-fixes
sudo apt-get install git linux-headers-generic build-essential dkms -y
git clone https://github.com/pvaret/rtl8192cu-fixes.git
sudo dkms add ./rtl8192cu-fixes
sudo dkms install 8192cu/1.11
sudo depmod -a
sudo cp ./rtl8192cu-fixes/blacklist-native-rtl8192.conf /etc/modprobe.d/

#echo "############### Resize Partition ###############" 
#cd jetcard
#git pull
#./scripts/jetson_install_nvresizefs_service.sh

sudo nmcli device wifi connect <SSID> password <PASSWORD>
sudo nvpmodel -m1

sudo reboot

