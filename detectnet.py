#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys,os
import base64
import argparse
import paho.mqtt.client as mqtt

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput, Log

x11window = videoOutput("display://0", None)

# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=detectNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--use_mqtt", action='store_true')
parser.set_defaults(use_mqtt=False)
    
try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# setup MQTT
if args.use_mqtt:
    print("Using MQTT")
    mqtt_client = mqtt.Client()
    mqtt_client.username_pw_set("<user>", "<pass>")
    try:
        mqtt_client.connect("<your_ip>", 0) # replace 0 with port
        print("Connected to MQTT broker service.")
    except:
        print("Failed to connect to MQTT server.")
    mqtt_client.loop_start() 
     
# create video sources and outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# load the object detection network
net = detectNet(args.network, sys.argv, args.threshold)

# note: to hard-code the paths to load a model, the following API can be used:
#
# net = detectNet(model="model/ssd-mobilenet.onnx", labels="model/labels.txt", 
#                 input_blob="input_0", output_cvg="scores", output_bbox="boxes", 
#                 threshold=args.threshold)

# process frames until EOS or the user exits
num = 0
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  
    
    # detect objects in the image (with overlay)
    detections = net.Detect(img, overlay=args.overlay)

    # stream to x11 window
    x11window.Render(img)
    x11window.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))
    num += 1

    if num % 25 == 0:        
        print(f"Processed {num} frames")
    
    if len(detections) > 0:
        # print the detections
        print("detected {:d} objects in image".format(len(detections)))

        for detection in detections:
            print(detection)

        # render the image to file on disk
        output.Render(img)

        # send to MQTT
        if len(detections) > 0 and args.use_mqtt and os.path.isfile(args.output):
            with open(args.output,'rb') as file:
                filecontent = base64.b64encode(file.read())
                #filecontent = file.read()
                byteArr = bytearray(filecontent)
                result = mqtt_client.publish("nvidia_jetson/delivery_detector/detection", byteArr)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    print("Image published to MQTT.")
                else:
                    print(f"Image not published to MQTT: {result.rc}")

    # exit on input/output EOS
    if not input.IsStreaming() or not x11window.IsStreaming():
        break

