import torch
from torchvision.transforms import functional as F
import numpy as np


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class ToTensor : 
    def __call__(self, image, mask) : 
        image = F.to_tensor(image)
        mask = torch.FloatTensor(np.array(mask).transpose(2,0,1))
        return image, mask