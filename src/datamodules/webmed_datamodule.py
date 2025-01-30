from .datasets.WebMed import WebMedDataset
from typing import Callable

from io import StringIO
import pandas as pd

import numpy as np


class WebNIH(WebMedDataset):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
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
        self.pathology_dict = dict(
            zip(self.pathologies, range(0, len(self.pathologies))))
        self.pathology_encoder = lambda x: self.pathology_dict[x]
        
        # Prepare webdataset stuff
        self.shards_url = list(
            filter(
                lambda x: "output" in x,
                (map(
                    lambda x: self.client.object_url(self.bucket, x["name"]),
                    self.client.list_objects(self.bucket),
                )),
            ))
        self.label_associator = self.get_associator()
        if self.only_bbox:
            self.selector = self.get_selector()

    def get_associator(self) -> Callable:
        # Get csv file
        raw_csv = self.client.get_object(self.bucket,
                                         "Data_Entry_2017_v2020.csv")
        data = StringIO(str(raw_csv, "utf-8"))
        self.csv = pd.read_csv(data)
        
        raw_csv = self.client.get_object(self.bucket,
                                         "BBox_List_2017.csv")
        data = StringIO(str(raw_csv, "utf-8"))

        self.bbox = pd.read_csv(data,names=["Image Index", "Finding Label", "x", "y", "w", "h", "_1", "_2", "_3"],
                                              skiprows=1)
        #Collect all masks together
        masks = list(map(lambda x: self.get_bbox(x[1]),self.bbox.iterrows()))
        print("b")

        d = dict()
        for i_id in masks:
            for items in i_id.items():
                if d.get(items[0]):
                    d[items[0]]["pathology_masks"].update(items[1]["pathology_masks"])
                else:
                    d[items[0]] = items[1]

        images = self.csv["Image Index"].str.replace(".png", "")
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

    def get_selector(self) -> Callable:
        select_images = self.bbox["Image Index"].str.replace(".png", "")
        return lambda x: select_images.str.contains(x["__key__"]).any()

class WebCheXpert(WebMedDataset):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        pathologies = [
            "Enlarged Cardiomediastinum",
            "Cardiomegaly",
            "Lung Opacity",
            "Lung Lesion",
            "Edema",
            "Consolidation",
            "Pneumonia",
            "Atelectasis",
            "Pneumothorax",
            "Pleural Effusion",
            "Pleural Other",
            "Fracture",
            "Support Devices",
        ]
        self.pathologies = sorted(pathologies)
        self.pathology_dict = dict(zip(pathologies, range(0,
                                                          len(pathologies))))
        self.pathology_encoder = lambda x: self.pathology_dict[x]

        # Prepare webdataset stuff
        self.shards_url = list(
            filter(
                lambda x: "wd" in x,
                (map(
                    lambda x: self.client.object_url(self.bucket, x["name"]),
                    self.client.list_objects(self.bucket),
                )),
            ))
        self.label_associator = self.get_associator()

    @classmethod
    def get_webdataset(
        cls,
        urls,
        label_associator,
        transform,
        target_transform,
        image_handler,
        selector,
    ):
        return super().get_webdataset(
            urls,
            label_associator,
            transform,
            target_transform,
            img_type="jpg",
            image_handler=image_handler,
            selector=selector,
        )

    def get_associator(self) -> Callable:

        def path_to_name(path):
            _, mode, patient, study, view = path.split("/")
            study_num = "".join(filter(str.isdigit, study))
            new_name = patient + "_" + study_num + "_" + view
            return new_name.replace(".jpg", "")

        # Get csv file
        csv = pd.read_csv(
            StringIO(
                str(self.client.get_object(self.bucket, "train.csv"),
                    "utf-8")))
        csv2 = pd.read_csv(
            StringIO(
                str(self.client.get_object(self.bucket, "valid.csv"),
                    "utf-8")))
        self.csv = pd.concat([csv, csv2])

        images = self.csv['Path'].apply(path_to_name).values
        self.labels = self.csv[self.pathologies].replace(-1,0).fillna(0).values

        # labels = list(range(0,len(images)))
        age = self.csv["Age"].values
        gender = (self.csv["Sex"] == "Male").values
        # labels = list(range(0,len(images)))
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
        return lambda x: self.associator[x]
