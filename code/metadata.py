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

description = vars(metadata[:20]["user_object"])["description"]
# words = description.split(" ");
# #words = [word if word.isalpha() for word in words]
# for word in words:
#     if word.isalpha():
#         print(word)
#     else:
#         print("no")
pp.pprint(description)
