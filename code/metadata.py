import math
import numpy as np
import os
import pickle as cPickle
import pprint 
import json
pp = pprint.PrettyPrinter(indent=2)

with open("../LLD-logo_metadata.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

myObject = metadata["beautiful-sail"]["user_object"]
jsonStr = json.dumps(myObject.__dict__)
pp.pprint(jsonStr["description"])
