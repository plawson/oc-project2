#!/usr/bin/env bash
sudo yum update -y
sudo yum install -y python34-setuptools
sudo easy_install-3.4 pip

sudo /usr/local/bin/pip3 install keras
sudo /usr/local/bin/pip3 install tensorflow
sudo /usr/local/bin/pip3 install h5py
sudo /usr/local/bin/pip3 install pillow
sudo /usr/local/bin/pip3 install boto3
sudo /usr/local/bin/pip3 install matplotlib
