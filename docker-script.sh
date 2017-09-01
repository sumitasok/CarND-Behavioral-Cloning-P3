#!/bin/sh
docker run -it --rm -p 8888:8888 -v `pwd`:/src -v ~/ml_data/Self-Driving-Car/Behavioural-Cloning/data:/data udacity/carnd-term1-starter-kit
