#!/bin/bash

# this downloads the hdf5 file that contains the data
curl https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-logo.hdf5 --output LLD-logo.hdf5

# this downloads the zip file that contains the metadata
curl https://data.vision.ee.ethz.ch/sagea/lld/data/LLD-logo_metadata.zip --output LLD-logo_metadata.zip
# this unzips the zip file
unzip LLD-logo_metadata.zip
# this cleans up the zip file, as we will no longer use it
rm LLD-logo_metadata.zip

echo downloaded data