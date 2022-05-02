import math
import numpy as np
import os
import pickle as cPickle

with open("LLD-logo_metada.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

print(metadata[:5])