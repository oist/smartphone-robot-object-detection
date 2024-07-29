This repo holds the python code for retraining the OIST smartphone robot object detector model.
This model is used to detect other robots and charging pucks.  

This project originally made use of [tf-model-maker](https://github.com/tensorflow/examples) but
currently there are a LOT of existing issues. A clean install with latest dependencies does
 NOT work. So, this project will maintain a simple Dockerfile to hold the required working 
dependencies to allow retraining until the tf-model-maker project starts working again, or 
this project moves to the potential offshoot project called [Media Pipe](https://ai.google.dev/edge/mediapipe/solutions/model_maker)

# Dependencies
1. docker
2. [Nvida container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#id1)

# Retrain model
`docker compose up --build`
Note the default command is to run the retraining script (main.py).
This takes all label data from ./xml_dataset, which is generated via labelImg (see below) 
and retrains a efficientdet_lite0 model. The xml_dataset contains relative paths to all image files 
located in ./image_set, bounding box, and label data. The exported model is saved as model.tflite
in the root of the project.

# Data Labeling
The data labeling was done using [labelImg](https://github.com/HumanSignal/labelImg) but this project is no longer maintained
and the follow up orject [Label Studio](https://github.com/HumanSignal/label-studio)
did not provide a simple way to reuse the dataset created within labelImg. 
The existing install instructions for labelImg did not work on Ubuntu >22.04, so a 
docker container was created to run the labeling tool. The Dockerfile is hosted on a fork 
of the original project [here](https://github.com/topherbuckley/labelImg)

The labeling will likely need to be updated to use a more modern tool, but for now the
labelImg tool is sufficient for the small dataset used in this project.
