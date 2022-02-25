from typing import Optional

from aistore.client import Client, Bck
import webdataset as wds
from pytorch_lightning import LightningDataModule

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split

import numpy as np


class WebMedDataset(LightningDataModule):
    def __init__(
        self,
        # general part
        client_url: str,
        bucket_name: str,
        transform=None,
        target_transform=None,
        data_aug=None,
        selector=None,
        image_handler="pil",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        seed=42,
        test_split=0.8,
        # dataset specific part
        unique_patients=True,
        views=["PA"],
        pathology_masks=False,
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        if transform:
            self.transform = transform
        else:
            self.transform = lambda x: x
        self.data_aug = data_aug
        self.image_handler = image_handler
        if target_transform:
            self.target_transform = target_transform
        else:
            self.target_transform = lambda x: x

        if selector:
            self.selector = selector
        else:
            self.selector = lambda x: True
        self.seed = seed
        self.test_split = test_split
        self.pin_memory = pin_memory

        self.bucket = Bck(bucket_name)
        self.client = Client(client_url)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`,
        `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()`
         and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether
        it's called before trainer.fit()` or `trainer.test()`."""
        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset_size = len(self.shards_url[:-1])
            test_size = int(self.hparams.test_split * dataset_size)
            train_size = dataset_size - test_size

            train_urls, test_urls = train_test_split(
                self.shards_url[:-1],
                test_size=self.hparams.test_split,
                random_state=self.hparams.seed,
            )
            val_urls = [self.shards_url[-1]]
            print(len(train_urls))

            self.data_train = self.get_webdataset(
                train_urls,
                self.label_associator,
                self.transform,
                self.target_transform,
                image_handler=self.image_handler,
                selector=self.selector,
            )
            self.data_test = self.get_webdataset(
                test_urls,
                self.label_associator,
                self.transform,
                self.target_transform,
                image_handler=self.image_handler,
                selector=self.selector,
            )
            self.data_val = self.get_webdataset(
                val_urls,
                self.label_associator,
                self.transform,
                self.target_transform,
                image_handler=self.image_handler,
                selector=self.selector,
            )

    @classmethod
    def get_webdataset(
        cls,
        urls,
        label_associator,
        transform,
        target_transform,
        selector=lambda x: True,
        img_type="png",
        image_handler="pil",
    ):
        len_epoch = len(urls)
        print(len_epoch)
        return (
            wds.WebDataset(urls)
            .decode(image_handler)
            .associate(label_associator)
            .select(selector)
            .rename(img=img_type)
            .map_dict(**{"img": transform, "label": target_transform})
        )

    def relabel_dataset(self, pathologies, silent=False):
        """
        Reorder, remove, or add (nans) to a dataset's labels.
        Use this to align with the output of a network.
        """
        will_drop = set(self.pathologies).difference(pathologies)
        if will_drop != set():
            if not silent:
                print("{} will be dropped".format(will_drop))
        new_labels = []
        self.pathologies = list(self.pathologies)
        for pathology in pathologies:
            if pathology in self.pathologies:
                pathology_idx = self.pathologies.index(pathology)
                new_labels.append(self.labels[:, pathology_idx])
            else:
                if not silent:
                    print("{} doesn't exist. Adding nans instead.".format(pathology))
                values = np.empty(self.labels.shape[0])
                values.fill(np.nan)
                new_labels.append(values)
        new_labels = np.asarray(new_labels).T

        self.labels = new_labels
        self.pathologies = pathologies
        self.associator = {
            kv[0]: kv[1] | {"label": nv}
            for kv, nv in zip(self.associator.items(), new_labels)
        }

    @property
    def num_classes(self) -> int:
        return len(self.pathology_dict)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
