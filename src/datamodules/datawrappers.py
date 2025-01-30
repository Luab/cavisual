import os
import numpy as np

from torchsampler import ImbalancedDatasetSampler
from torchxrayvision.datasets import VinBrain_Dataset,NIH_Dataset,relabel_dataset,CheX_Dataset
from torchvision.transforms import Compose
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer, normalize, apply_transforms
from skimage.io import imread

class NIH_wrapper(NIH_Dataset):
    def __getitem__(self,idx):
        res = super().__getitem__(idx)
        res['label'] = res['lab']
       # res['meta'] = {"age":np.array(self.csv["Patient Age"].iloc[idx]),
       #                 "gender":np.array((self.csv["Patient Gender"] == "M").iloc[idx])}
        return res

class VINBig_wrapper(VinBrain_Dataset):
    def __getitem__(self,idx):
        res = super().__getitem__(idx)
        res['label'] = res['lab']
        return res


class CheX_wrapper(CheX_Dataset):
    def __getitem__(self,idx):
        res = super().__getitem__(idx)
        res['label'] = res['lab']
        return res
