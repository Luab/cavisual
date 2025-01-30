import os
import numpy as np
import torch
import random

from torchxrayvision.datasets import VinBrain_Dataset,NIH_Dataset,CheX_Dataset
from torchvision.transforms import Compose
from torchxrayvision.datasets import normalize
from skimage.io import imread

def apply_transforms(sample, transform, seed=None) -> dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is not None:
        random.seed(seed)
        torch.random.manual_seed(seed)
        sample["img"] = transform(sample["img"])

        #TODO:Implement
        #if "pathology_masks" in sample:
        #    for i in sample["pathology_masks"].keys():
        #        random.seed(seed)
        #        torch.random.manual_seed(seed)
        #        sample["pathology_masks"][i] = transform(sample["pathology_masks"][i])

        if "semantic_masks" in sample:
            for i in sample["semantic_masks"].keys():
                random.seed(seed)
                torch.random.manual_seed(seed)
                sample["semantic_masks"][i] = transform(sample["semantic_masks"][i])

    return sample
class NIH_wrapper(NIH_Dataset):

    def __init__(self,
                 imgpath,
                 csvpath=None,
                 bbox_list_path=None,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True,
                 pathology_masks=False) -> None:
        super().__init__(imgpath,
                 csvpath,
                 bbox_list_path,
                 views,
                 transform,
                 data_aug,
                 nrows,
                 seed,
                 unique_patients,
                 pathology_masks)
        images = self.csv["Image Index"].str.replace(".png", "")
        age = self.csv["Patient Age"].values
        gender = (self.csv["Patient Gender"] == "M").values

        masks = list(map(lambda x: self.get_bbox(x[1]),self.pathology_maskscsv.iterrows()))

        d = dict()
        for i_id in masks:
            for items in i_id.items():
                if d.get(items[0]):
                    d[items[0]]["pathology_masks"].update(items[1]["pathology_masks"])
                else:
                    d[items[0]] = items[1]


        self.associator = dict(
            zip(
                images,
                map(
                    lambda x: {
                        "label": x[0]
                        #,
                       # "meta": {
                        #    "age": x[1],
                        #    "gender": x[2]
                        #}
                    },
                    zip(self.labels, age, gender),
                ),
            ))
        # Add masks to associator
        self.associator = {k: {**v, **d.get(k,{})} for k,v in self.associator.items()}

    def __getitem__(self,idx):
        sample = {}
        sample["idx"] = idx

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = sample | self.associator[imgid.replace(".png", "")]
        sample['imgid'] = imgid
        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def __len__(self):
        return len(self.csv)

    def get_bbox(self,row,this_size=224) -> dict:
            scale = this_size / 1024
            key = row.str.replace(".png", "")
            mask = np.zeros([this_size, this_size])
            xywh = np.asarray([row.x, row.y, row.w, row.h])
            xywh = xywh * scale
            xywh = xywh.astype(int)
            mask[xywh[1]:xywh[1] + xywh[3], xywh[0]:xywh[0] + xywh[2]] = 1

            # Resize so image resizing works
            mask = mask[None, :, :]
            return {key.iloc[0]:{"pathology_masks":{row["Finding Label"]:{"mask":mask,"mask_label":row["Finding Label"]}}}}


class VINBig_wrapper(VinBrain_Dataset):
    def __getitem__(self,idx):
        res = super().__getitem__(idx)
        res['label'] = res['lab']
        return res


class CheX_wrapper(CheX_Dataset):
    def a(self) -> None:
        pass