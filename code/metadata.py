import math
import numpy as np
import os
import pickle as cPickle
import pprint 
import json
import re
pp = pprint.PrettyPrinter(indent=2)

with open("../LLD-logo_metadata.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

description = vars(metadata["beautiful-sail"]["user_object"])["description"]
words = description.split();
words = [word if word.isalpha for word in words]
pp.pprint(words)
