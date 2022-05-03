import math
import numpy as np
import os
import pickle as cPickle
import pprint 
import json
import re
from googletrans import Translator
pp = pprint.PrettyPrinter(indent=2)

with open("../LLD-logo_metadata.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

print(list(metadata.keys())[:100])
description = vars(metadata["google"]["user_object"])["description"]
print(metadata["dna"]["user_object"])
# translator = Translator()
# print(type(description))
# translated = translator.translate(str(description))
# words = description.split(" ");
# #words = [word if word.isalpha() for word in words]
# for word in words:
#     if word.isalpha():
#         print(word)
#     else:
#         print("no")
#pp.pprint(description)
#pp.pprint(translated)
