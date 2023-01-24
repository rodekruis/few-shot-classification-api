#!/bin/bash

cd /home/huggingface-user/text-classification-app || return
sudo git pull
sudo docker system prune -a -f
sudo docker build -t text-classification-app .
sudo docker run --name text-classification-app --gpus all -p 80:8000 -t text-classification-app

# systemd service specification
#[Unit]
#Requires=docker.service
#After=docker.service
#
#[Service]
#TimeoutStartSec=0
#KillMode=none
#Restart=always
#RestartSec=5s
#ExecStart=/home/huggingface-user/text-classification-app/docker-build-run.sh
#
#[Install]
#WantedBy=default.target