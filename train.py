'''
Train an Electra-based classifer on the IMDB dataset
'''

import torch
from transformers import ElectraTokenizer
from data_prep import get_train, get_val
