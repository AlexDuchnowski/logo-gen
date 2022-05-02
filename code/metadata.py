import math
import numpy as np
import os
import pickle as cPickle

with open("LLD-icon-names.pkl", 'rb') as f:
    icon_names = cPickle.load(f, encoding='latin1')

print(icon_names[:5])