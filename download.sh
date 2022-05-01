#!/bin/bash

#this downloads the zip file that contains the data
curl https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-icon_PKL.zip --output LLD-icon_PKL.zip
# this unzips the zip file
unzip LLD-icon_PKL.zip
# this cleans up the zip file, as we will no longer use it
rm LLD-icon_PKL.zip

echo downloaded data