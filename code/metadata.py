import math
import numpy as np
import os
import pickle as cPickle
import pprint as pp
pp = pprint.PrettyPrinter(indent=2)

with open("../LLD-logo_metadata.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

print(type(metadata["060608"]));
pp.pprint(metadata["060608"]["user_object"]);
pp.pprint(metadata["060608"]["user_object"]["description"]);
