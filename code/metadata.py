import math
import numpy as np
import os
import pickle as cPickle
import pprint 
pp = pprint.PrettyPrinter(indent=2)

with open("../LLD-logo_metadata.pkl", 'rb') as f:
    metadata = cPickle.load(f, encoding='latin1')

pp.pprint(metadata["beautiful-sail"]["user_object"]);
pp.pprint(type(metadata["beautiful-sail"]["user_object"]));
pp.pprint(metadata["beautiful-sail"]["user_object"].get_description());
pp.pprint(metadata["beautiful-sail"]["user_object"].getDescription());
pp.pprint(metadata["beautiful-sail"]["user_object"].get("description"));
