import math
import numpy as np
import os
import pickle as cPickle

with open("../LLDsample/LLD-logo_metadata.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

print(metadata["user_object"])
