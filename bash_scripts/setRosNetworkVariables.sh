#!/bin/bash


export ROS_IP=$(/bin/hostname -i)
export ROS_HOSTNAME=$(/bin/hostname -i)
export ROS_MASTER_URI=http://$1:11311/
