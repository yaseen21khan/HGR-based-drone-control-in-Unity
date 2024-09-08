# HGR Based Drone Control in Unity

This repository contains Unity executables for simulating drone control using hand gesture recognition (HGR). The system uses a Python program to recognize hand gestures and send control commands to the drone simulation over sockets.

## System Overview

- **Unity Executables**: Simulate a drone that waits for control commands such as moving left, right, up, down, forward, backward, take off, land, etc.
- **Python Program**: Recognizes hand gestures via an RGB camera and sends commands to the drone using socket communication.
- **IP Configuration**: The IP addresses in the Python and Unity scripts must be updated to match your machine's local IP address.
- **Hardware Requirement**: An RGB camera must be connected to capture gestures for recognition.

## Setup Instructions

1. Clone this repository to your local machine.
2. Update the IP addresses in both the Unity and Python scripts to match your local network setup.
3. Connect an RGB camera to the system.
4. Run the Python hand gesture recognition script to begin sending control signals to the Unity drone simulator.
5. Launch the Unity executable to start the drone simulation.

## Requirements

- Unity 3D Engine
- Python 3.x
- OpenCV, Mediapipe, TensorFlow for hand gesture recognition
- RGB Camera

## License
See LICENSE.txt
