from typing import Callable

import h5py
import pandas as pd
import numpy as np

import torch
from torch.utils import data


class CXRDataset(data.Dataset):
    """Represents an abstract HDF5 dataset.
    
    Input params:
        file_path: Path to the folder containing the dataset (one or multiple HDF5 files).
        recursive: If True, searches for h5 files in subdirectories.
        load_data: If True, loads all the data immediately into RAM. Use this if
            the dataset is fits into memory. Otherwise, leave this at false and 
            the data will load lazily.
        data_cache_size: Number of HDF5 files that can be cached in the cache (default=3).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, img_path,file_path, size=None, transform=None, repeat=3,to_torch=True):
        super().__init__()
        if size != None: 
            self.img_dset = h5py.File(img_path, 'r')['cxr'][:size]
            self.file_path = pd.read_csv(file_path)['Path'][:size]
        else: 
            self.img_dset = h5py.File(img_path, 'r')['cxr']
            self.file_path = pd.read_csv(file_path)['Path']
        self.transform = transform
        self.repeat = repeat
        self.to_torch = to_torch
            
    def __len__(self):
        return len(self.file_path)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = self.img_dset[idx] # np array, (320, 320)
        path = self.file_path[idx]
        img = np.expand_dims(img, axis=0)
        img = np.repeat(img, self.repeat, axis=0)

        if self.to_torch:
            img = torch.from_numpy(img) # torch, (3, 320, 320)
        if self.transform:
            img = self.transform(img)
        sample = {'img': img, "filepath":path}
        
        return sample


class MIMICDataset(CXRDataset):
    def __init__(self, img_path,file_path, txt_path, column='report', size=None, transform=None,repeat=3,to_torch=True):
        super().__init__(img_path, file_path, size, transform,repeat,to_torch)
        if size != None: 
            self.txt_dset = pd.read_csv(txt_path)[column][:size]
        else: 
            self.txt_dset = pd.read_csv(txt_path)[column]

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        txt = self.txt_dset[idx] # python str

        if type(txt) == type(float("nan")): # capture the case of empty "Impression" sections
            txt = " "

        return sample | {'txt': txt}

class NIHDataset(CXRDataset):
    def __init__(self, img_path, file_path,csv_path,bbox_path, size=None, transform=None,repeat=3,to_torch=True):
        super().__init__(img_path, file_path, size, transform,repeat,to_torch)
        self.pathologies = [
            "Atelectasis",
            "Consolidation",
            "Infiltration",
            "Pneumothorax",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Effusion",
            "Pneumonia",
            "Pleural_Thickening",
            "Cardiomegaly",
            "Nodule",
            "Mass",
            "Hernia",
            "No Finding",
        ]

        self.pathologies = sorted(self.pathologies)
        self.label_associator = self.get_associator(csv_path,bbox_path)
        ##TODO:Implement View filter 

        assert(len(self.associator) == len(self.img_dset))

    def __getitem__(self, idx):
        sample =  super().__getitem__(idx)
        filename = sample['filepath'].split("/")[-1]
        label_data = self.associator[filename]
        txt = self.label_to_txt(label_data)
        return sample | {"txt":txt}

    def label_to_txt(self,label_data):
        """
        Convert label data to prompt
        """
        return " ".join(np.array(self.pathologies)[label_data['label'].astype(bool)])

    def get_associator(self,csv_path,bbox_path) -> Callable:
        # Get csv file
        self.csv = pd.read_csv(csv_path)
        
        self.bbox = pd.read_csv(bbox_path,names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
                                            skiprows=1)
        #Collect all masks together
        masks = list(map(lambda x: self.get_bbox(x[1]),self.bbox.iterrows()))

        d = dict()
        for i_id in masks:
            for items in i_id.items():
                if d.get(items[0]):
                    d[items[0]]["pathology_masks"].update(items[1]["pathology_masks"])
                else:
                    d[items[0]] = items[1]

        images = self.csv["Image Index"]
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(
                self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        age = self.csv["Patient Age"].values
        gender = (self.csv["Patient Gender"] == "M").values

        self.associator = dict(
            zip(
                images,
                map(
                    lambda x: {
                        "label": x[0],
                        "meta": {
                            "age": x[1],
                            "gender": x[2]
                        }
                    },
                    zip(self.labels, age, gender),
                ),
            ))
        # Add masks to associator
        self.associator = {k: {**v, **d.get(k,{})} for k,v in self.associator.items()}
        return lambda x: self.associator[x]

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


class VinbigDataset(CXRDataset):
    def __init__(self, img_path, file_path,csv_path, size=None, transform=None,repeat=3,to_torch=True):
        super().__init__(img_path, file_path, size, transform,repeat,to_torch)
        self.pathologies = ['Aortic enlargement',
                            'Atelectasis',
                            'Calcification',
                            'Cardiomegaly',
                            'Consolidation',
                            'ILD',
                            'Infiltration',
                            'Lung Opacity',
                            'Nodule/Mass',
                            'Lesion',
                            'Effusion',
                            'Pleural_Thickening',
                            'Pneumothorax',
                            'Pulmonary Fibrosis']

        self.pathologies = sorted(np.unique(self.pathologies))

        self.mapping = dict()
        self.mapping["Pleural_Thickening"] = ["Pleural thickening"]
        self.mapping["Effusion"] = ["Pleural effusion"]
        self.rawcsv = pd.read_csv(csv_path)
        self.csv = pd.DataFrame(self.rawcsv.groupby("image_id")["class_name"].apply(lambda x: "|".join(np.unique(x))))
        self.csv["has_masks"] = self.csv.class_name != "No finding"
        self.label_associator = self.get_associator()
        assert(len(self.associator) == len(self.img_dset))

    def __getitem__(self, idx):
        sample =  super().__getitem__(idx)
        filename = sample['filepath'].split("/")[-1]
        label_data = self.associator[filename.replace(".dicom","")]
        txt = self.label_to_txt(label_data)
        return sample | {"txt":txt}

    def label_to_txt(self,label_data):
        """
        Convert label data to prompt
        """
        return " ".join(np.array(self.pathologies)[label_data['label'].astype(bool)])

    def get_associator(self) -> Callable:

        images = self.csv.reset_index()["image_id"]
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["class_name"].str.lower().str.contains(pathology.lower())
            if pathology in self.mapping:
                for syn in self.mapping[pathology]:
                    mask |= self.csv["class_name"].str.lower().str.contains(syn.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        associator = dict(
            zip(
                images,
                map(
                    lambda x: {
                        "label": x,
                    },
                    self.labels,
                ),
            ))
        # Add masks to associator
        self.associator = {k : v | {"pathology_masks" : self.get_mask_dict(k)} for k,v in associator.items()}

        return lambda x: self.associator[x]

    def get_mask_dict(self, image_name, this_size=(1,224,224)):

        c, h, w = this_size

        path_mask = {}
        rows = self.rawcsv[self.rawcsv.image_id.str.contains(image_name)]

        for i, pathology in enumerate(self.pathologies):
            for group_name, df_group in rows.groupby("class_name"):
                if (group_name == pathology) or ((pathology in self.mapping) and (group_name in self.mapping[pathology])):

                    mask = np.zeros([h, w])
                    for idx, row in df_group.iterrows():
                        mask[int(row.y_min):int(row.y_max), int(row.x_min):int(row.x_max)] = 1

                    path_mask[i] = mask[None, :, :]

        return path_mask