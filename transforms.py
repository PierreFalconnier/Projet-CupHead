import torch
from torchvision.transforms import ToTensor, Resize, Grayscale
from torch.nn import Threshold
from typing import Any, Dict, List, Tuple, Type, Callable, Optional
import random

class MyPilToTensor(object):
    """
    Convert a PIL image as a torch.Tensor

    Attributs:
        TrToTensor (class): Convert a PIL image as a torch.Tensor
    """
    def __init__(self)-> None:
        self.TrToTensor=ToTensor()

    def __call__(self,sample)-> dict:
        sample['x']= self.TrToTensor(sample['x'])
        return sample

class MyGrayscale(object):
    """
    Convert RGB image to grayscale image
    """
    def __init__(self) -> None:
        self.TrGrayscale = Grayscale()

    def __call__(self, sample) -> Any:
        sample["x"] = self.TrGrayscale(sample["x"])
        return sample
        
    

class MyResize(object):
    """
    Resize a tensor
    """
    def __init__(self, size_tupple=(84,84)) -> None:
        self.TrResize = Resize(size = size_tupple)


    def __call__(self, sample) -> dict:
        sample["x"] = self.TrResize(sample["x"])
        return sample
    
class MinMax(object):

    def __init__(self) -> None:
        pass 
    
    def __call__(self, sample) -> dict :
        if 'hsi_x' in sample:               # T C H W
            maxs = torch.amax(sample['hsi_x'], dim=(2,3)).unsqueeze(-1).unsqueeze(-1)   # T C 1 1
            mins = torch.amin(sample['hsi_x'], dim=(2,3)).unsqueeze(-1).unsqueeze(-1)
            sample['hsi_x'] = (sample['hsi_x']-mins)/(maxs-mins)                        # broadcasting
        return sample

class StackFeature(object):
    """
    Stack frames
    """
    def __init__(self,stack_size) -> None:
        self.stack_size = stack_size

    def __call__(self,sample) -> Any:

        pass
        
        return sample

