# Tensorflow Object Detector with ROS

This project extends a previous implementation of tensorflow object detection in ROS.
Using a depth camera, pixel coordinates are reprojected into 3D and published to /tf. 

## Requirements:

Tensorflow and ROS

This guide targets Ubuntu 16.04 and ROS Kinetic

## Steps:

Start the rosnode of your depth camera
Then,
`rosrun tensorflow_object_detector detect_ros_and_find_depth.py`

