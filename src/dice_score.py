import torch
from torch import Tensor
import numpy as np
def dicescore(inputs, targets):
    intersection = np.sum(np.multiply(inputs, targets))
    dice = 2*intersection / (np.sum(inputs) + np.sum(targets))
    return dice

def sensitivity(inputs, targets) : 
    num = np.sum(np.multiply(inputs, targets))
    denom = np.sum(targets)
    return 1 if denom == 0 else num / denom

def specificity (inputs, targets):
	#computes false positive rate
	num=np.sum(np.multiply(targets==0, inputs==0))
	denom=np.sum(targets==0)
	return 1 if denom == 0 else num / denom