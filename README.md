# mailmon
Package delivery vehicle detection and alerting through Home Assistant integrations

## Quickstart for the impatient

* Buy an [NVIDIA Jetson Orin Nano device](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
* Flash it with the stock image according to the instructions in the getting started tutorial on Jetson website
* Follow the setup instructions for the [Jetson inference repository](https://github.com/dusty-nv/jetson-inference/tree/master)
* Copy these files from this repository to any location on the Jetson device:
  * `detectnet.py`
  * `labels.txt`
  * `ssd-mobilenet.onnx` (the fine-tuned model extracted from `ssd-mobilenet.zip`)
  * `test_dataset.zip\*.*` (these are test images to use, if needed)
  * Edit `detectnet.py` to specify your HASS IP, user name and password.
* Run this command:
```python detectnet.py --model=ssd-mobilenet.onnx --labels=labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes --confidence=.75 --use_mqtt <your_RTSP_URI> result.jpg```

You will want to play with `--confidence`.

To train with your own dataset, continue reading.

## Motivation
My motivations for this project were two-fold:
* familiarize myself with my recently-purchased [NVIDIA Jetson Orin Nano device](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
* increase my exposure to fine-tuning foundation models using NVIDIA's software stack

But what model did I want to use and how was I going to get training data?  Well, for a while now, I've been wanting to solve a problem close to home: knowing when the mail delivery person arrives. It occurred to me that streaming my gate camera video feed to the Jetson, where I could run inference in real-time, might be the ticket.  Fortunately, there are some [fantastic tutorials on Jetson inference](https://github.com/dusty-nv/jetson-inference/tree/master), written by Nvidia's own [Dustin Franklin](https://github.com/dusty-nv), that do just that.

It's worth noting that there is at least [one project](https://github.com/JoeTester1965/CudaCamz) out there that is very similar to my project, except CudaCamz runs directly on SSD-Mobilenet-v2. I chose to fine-tune this model instead.

## Getting started

There's no need for me to cover all the pre-requisite steps, like setting up Jetson JetPack.  Dusty does a fantastic job of that in the [Hello AI World tutorial](https://github.com/dusty-nv/jetson-inference/tree/master#hello-ai-world).

The only thing I needed to do was configure my Unifi cameras to stream RTSP from the Protect app:
 * Select the camera from Unifi Devices dashboard
 * Click Settings
 * Click Advanced
 * Under Real Time Streaming Protocol (RTSPS), select the desired resolution of High, Medium or Low by ticking the checkbox.

 Once the checkbox has been ticked, a URL will be displayed like `rtsps://192.168.1.2:7441/wPXqmXadtJmll8FK?enableSrtp`. To use the unencrypted version, convert this link to `rtsp://192.168.1.2:7447/wPXqmXadtJmll8FK`

 The tutorial scripts support this streaming protocol right out of the box. You can test it with the `video-viewer` tool. Read [this page](https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md) for more details.

## Object detection

Because I wanted to recognize a delivery truck/van, I focused on the [object detection tutorial](https://github.com/dusty-nv/jetson-inference/blob/master/docs/detectnet-console-2.md). The tutorial was very straightforward and I was running the SSD-Mobilenet-v2 model on my RTSP stream in about an hour.

When developing your own ML project, one thing to consider if you're not sure which model is best for your use case:  try out [NVIDIA's Auto ML](https://developer.nvidia.com/blog/training-like-an-ai-pro-using-tao-automl/).

## Fine-tuning

The SSD-Mobilenet-v2 model has a limited set of labels/classes, and for vehicles, only understands "truck" and "car". NVIDIA's model catalog (NGC) contained another promising model called [VehicleMakeNet](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/vehiclemakenet), which is a classification model based on Resnet18, and understands a wealth of vehicle makes and models. However, this does not help with classifying delivery trucks, and because it's a classification model and not an object detection one, I would need to learn how to "cascade" the two in order to achieve my goals. That would require learning the details of either NVIDIA TAO or DeepStream, which I didn't feel like tackling given my rather simple goals.

In the search for some open source dataset that might fit the bill (such as [openimages](https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=detection&c=%2Fm%2F0k4j) or [Stanford's car dataset](https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset)), I came across [Roboflow's logoimages dataset](https://universe.roboflow.com/capstoneproject/logoimages) which contained classifications for all the major delivery services (USPS, FedEx, UPS and Amazon) based on both logo and vehicle type. This seemed perfect!

The [tutorial for fine-tuning SSD](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-ssd.md) described the basic steps, but I found it a bit troublesome to get the VOC XML format correct. When I exported the dataset from Roboflow in VOC XML format, it was not in the directory structure required by the tutorial's scripts (as others have noted [1](https://forums.developer.nvidia.com/t/having-trouble-setting-up-pytorch-code-for-training-ssd-mobilenet/188199/5), [2](https://forums.developer.nvidia.com/t/dusty-nv-jetson-training-custom-data-sets-generating-labels/175008/9)). So, I had to manually copy the files into the right places and create the various `train.txt`, `val.txt`, `trainval.txt`, etc. Roughly, the steps involved:
* Create the required directory structure:
  * `Annotations`
  * `ImageSets`
    * `Main`
      * `test.txt` - contains the file names (no extension) of only the test images
      * `train.txt` - contains the file names (no extension) of only the training images
      * `trainval.txt` - contains the file names (no extension) of the training and validation images
      * `val.txt` - contains the file names (no extension) of only the validation images
  * `JPEGImages` - contains all of the JPG images from training, validation and test data sets
* Copy the directory structure to the `jetson-inference/python/training/detection/ssd/data/delivery` directory
* Train for 30 epochs `python train_ssd.py --dataset-type=voc --data=data/delivery --model-dir=models/delivery --batch-size=4 --epochs=30`
* Convert to ONNX for inference: `python3 onnx_export.py --model-dir=models/delivery`
* Test a few images not included during training or validation:
	```lilhoser@whiteoak-jetson:~/Downloads/jetson-inference/python/training/detection/ssd$ detectnet.py --model=models/delivery/ssd-mobilenet.onnx --labels=models/delivery/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes ~/Downloads/jetson-inference/python/training/detection/ssd/data/test/USPS50_jpeg.rf.3098990dba9f965ee82317afd1679e4e.jpg result.jpg```
* Run live inference on camera stream:
    ```lilhoser@whiteoak-jetson:~/Downloads/jetson-inference/python/training/detection/ssd$ detectnet --model=models/delivery/ssd-mobilenet.onnx --labels=models/delivery/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes rtsp://192.168.1.2:7447/wPXqmXadtJmll8FK display://0```

### A word about datasets

It's a fact that obtaining high-quality, diverse training data is one of the trickiest problems of applied ML today. It seems that the common recommendation, as is [discussed in the tutorial](https://github.com/dusty-nv/jetson-inference/blob/master/docs/pytorch-collect-detection.md) is to manually generate the images. This process, while certainly educational and informative, seems incredibly tedious and time consuming, with extremely limited results. Some techniques such as data augmentation seem promising (for example, automatically adjusting hue, saturation, brightness, and so on) and are a standard feature of most tooling and frameworks like [TAO](https://developer.nvidia.com/tao-toolkit-usecases-whitepaper/3-quick-prototyping-small-dataset#32-What-is-data-augmentation) and [Edge Impulse](https://docs.edgeimpulse.com/docs/tips-and-tricks/data-augmentation). Going a step further, it's possible to generate synthetic training data using tools like [NVIDIA's Omniverse replicator, Isaac Slim, or Drive Sim](https://www.nvidia.com/en-us/omniverse/synthetic-data/). There are also clever platforms out there like [Edge Impulse](https://docs.edgeimpulse.com/docs/edge-impulse-studio/data-acquisition) that provide convenient ways to automatically upload and polish your data (whether it be images, audio clips, or whatever) before training.

## Home Assistant Integration

Now that the fine-tuned model appears to be working on my live camera stream, my next objective was to add an alerting capability. My house is rigged with various home automation gadgets and most (but not all) of them are wired into [Home Assistant (HASS)](https://www.home-assistant.io/getting-started) integrations. I've already created a service workflow in HASS for iPhone alerts, so I figured it could be quite simple to forward image detections from my NVIDIA Jetson device to HASS and onto my phone. The workflow will be:

* Publish the binary image to an MQTT topic from the NVIDIA Jetson device
* Create an MQTT camera integration to handle the incoming image
* Create an automation that:
  * listens to the topic and forwards any received binary image to the MQTT camera
    * the MQTT camera will take a snapshot and write it to disk
  * calls the mobile notify service to send a message to my iPhone with the URL of the image

### Setup MQTT camera integration

The [MQTT camera integration](https://www.home-assistant.io/integrations/camera.mqtt/) is built into HASS, so activating is a few extra lines to the `configuration.yaml` file:
```
mqtt:
    camera:
        - topic: nvidia_jetson/delivery_detector/detection
```

Soft restart HASS.

### Add a card to the dashboard (optional)

For testing purposes, it's useful to see that the image is being received by the MQTT camera integration. This can be accomplished with a new card on the HASS dashboard:
* Edit dashboard
* Add card - picture entity card
  * Entity: MQTT Camera (`camera.mqtt_camera`)
  * image path: set to empty

Whenever a detection image is sent by the NVIDIA Jetson device, you can view it in this dashboard card.

### Configure an MQTT automation

Navigate to Settings -> Automations and Scenes and click Create Automation.  Edit in YAML and add this code:

```
alias: Nvidia Jetson Delivery Vehicle Alert
description: >-
    Sends an iphone alert whenever the Nvidia Jetson device detects a delivery
    vehicle on the road.
trigger:
    - platform: mqtt
    topic: nvidia_jetson/delivery_detector/detection
    encoding: ""
condition: []
action:
    - variables:
        fname: gate_snapshot_{{ now().strftime("%I%M_%p_%Y-%m-%d")}}.jpeg
    - service: camera.snapshot
    data:
        filename: /config/www/nvidia_jetson_photos/{{fname}}
    target:
        entity_id: camera.mqtt_camera
    - alias: Wait 2s
    delay: 2
    - service: notify.mobile_app_aaron_s_iphone
    data:
        message: A delivery vehicle has been detected.
        title: Delivery Vehicle Alert!
        data:
        image: /local/nvidia_jetson_photos/{{fname}}
mode: single
```

A few important notes that took me several hours, searching and investigating to figure out:

* It's important to set the encoding to an empty string on the trigger, otherwise HASS will try and fail to decode binary image data
* We're writing the image to the HASS web server because the other mechanism of writing it to an external folder and then allow-listing that folder into the HASS docker image is chaotic.
* An arbitrary 2-second wait attempts to allow the filesystem enough time to write the image to disk before calling the alert service
* The `/local` path in the iphone alert is intentional and necessary. HASS does an URL rewrite to convert this virtual/relative path to an actual URL.

### Update the `detectnet.py` script

The tutorial's script, `detectnet.py`, by default just writes the detection image to disk or to the X11 window/display. The modified version I have included in this repository additionally transmits the image to the HASS device (which is a pi4b device sitting in my server rack) by publishing it to the required MQTT topic. Note that the image data is base64-encoded, but this is probably not necessary.

### Testing

A simple test of the modified `detectnet.py` script using one of the dataset's test image:

```lilhoser@whiteoak-jetson:~/Downloads/jetson-inference/python/training/detection/ssd$ python ~/Downloads/detectnet.py --model=models/delivery/ssd-mobilenet.onnx --labels=models/delivery/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes --use_mqtt ~/Downloads/jetson-inference/python/training/detection/ssd/data/test/USPS50_jpeg.rf.3098990dba9f965ee82317afd1679e4e.jpg result.jpg```

This image should contain detection labeling and bounding boxes, and it should also show up in the HASS dashboard card as well as an iphone alert.

Testing on live stream:

```lilhoser@whiteoak-jetson:~/Downloads/jetson-inference/python/training/detection/ssd$ python ~/Downloads/detectnet.py --model=models/delivery/ssd-mobilenet.onnx --labels=models/delivery/labels.txt --input-blob=input_0 --output-cvg=scores --output-bbox=boxes --use_mqtt rtsp://192.168.1.2:7447/wPXqmXadtJmll8FK result.jpg```

## What now?

Despite it all working quite well, I've actually had limited success with the accuracy of this fine-tuned model on my own livestream, probably due to a weak training dataset. Differences in lighting and objects in my camera's field of view could be the culprit. My next step is to try data augmentation using TAO and Edge Impulse toolkits. If that does not work, I plan to look into synthetic data generation.

I hope to update the repository soon with a more precise fine-tuned model.

## References and related reading

* https://github.com/robmarkcole/HASS-Deepstack-object
* https://github.com/johnolafenwa/DeepStack
* https://forums.developer.nvidia.com/t/how-i-used-jetson-nano-and-vertex-ai-to-catch-a-bus/225846
* https://github.com/JoeTester1965/CudaCamz
* https://github.com/steveseguin/raspberry_ninja/tree/main