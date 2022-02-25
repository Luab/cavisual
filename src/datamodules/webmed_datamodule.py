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

    def get_associator(self) -> Callable:
        # Get csv file
        raw_csv = self.client.get_object(self.bucket,
                                         "Data_Entry_2017_v2020.csv")
        data = StringIO(str(raw_csv, "utf-8"))
        self.csv = pd.read_csv(data)

        images = self.csv["Image Index"].str.replace(".png", "")
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(
                self.csv["Finding Labels"].str.contains(pathology).values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        age = self.csv["Patient Age"].values
        gender = (self.csv["Patient Gender"] == "M").values
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

        images = self.csv.Path.apply(path_to_name).values
        self.labels = self.csv[self.pathologies].values
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
