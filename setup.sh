#!/usr/bin/env bash

mkdir data
pushd data
echo "Downloading synthetic dataset ..."
curl http://mrg.robots.ox.ac.uk:8080/MRGData/deeptracking/DeepTracking_1_1.t7.zip -o deep-tracking.zip
echo "Extracting synthetic dataset ..."
unzip deep-tracking.zip
popd
