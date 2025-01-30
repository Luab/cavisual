# %%
import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
import matplotlib.pyplot as plt
import numpy as np

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torchmetrics.regression import CosineSimilarity
import torchxrayvision as xrv 
import os, sys
sys.path.append("..")

from src import utils
import pandas as pd
import wandb
import skimage, skimage.filters
import sklearn, sklearn.metrics


from utils import generate_explanation,generate_vector,generate_vector_cav,calc_iou

hydra.initialize(config_path="../configs")
config=hydra.compose(config_name="config.yaml",overrides=["experiment=example"])

os.environ['WANDB_NOTEBOOK_NAME'] = "gifsplanation_playground.ipynb"


# %%
from typing import List
from kmeans_pytorch import kmeans
from einops import rearrange

class ClusteredVectorFinder:
    def __init__(self,num_clusters: int, vectors: torch.tensor, distance='cosine') -> None:
        vectors = rearrange(vectors, "num a b c d -> num (a b c d)")
        self.num_clusters = num_clusters
        self.distance = distance
        self.cluster_ids_x, self.cluster_centers = kmeans(
            X=vectors, num_clusters=num_clusters, distance=distance, device=torch.device('cuda:0')
        )


    def find_closest(self,vector):
        dist = torch.cosine_similarity(self.cluster_centers,vector.to("cpu").reshape(1,-1))
        min_index = torch.argmin(dist)
        return self.cluster_centers[min_index].to("cuda").view_as(vector)

# %%
ae = xrv.autoencoders.ResNetAE(weights="101-elastic")
clf = xrv.models.DenseNet(weights="all")
ae = ae.cuda()
clf = clf.cuda()

#Target dataset
config.datamodule.bucket_name = "nih_bbox"
config.datamodule.test_split = 0
config.datamodule.batch_size = 1

NIH_datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
NIH_datamodule.relabel_dataset(clf.pathologies,silent=False)
NIH_datamodule.setup()

# %%
from torchsampler import ImbalancedDatasetSampler
from torchvision.transforms import Compose
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer, normalize, apply_transforms, relabel_dataset
from datawrappers import NIH_wrapper, CheX_wrapper

transforms = Compose([XRayCenterCrop(),XRayResizer(224)])
nih_ds = NIH_wrapper("/home/luab/experiments/data/nih_raw/",\
    csvpath="/home/luab/experiments/data/nih_raw/Data_Entry_2017_v2020.csv",\
    bbox_list_path="/home/luab/experiments/data/nih_raw/BBox_List_2017.csv",\
    unique_patients=True,
    transform=transforms)
relabel_dataset(clf.pathologies,nih_ds,silent=False)

chex_ds = CheX_wrapper("~/experiments/data/chexpert_raw","~/experiments/data/chexpert_raw/train.csv",transform=transforms)
relabel_dataset(clf.pathologies,chex_ds,silent=False)
chex_ds.labels = np.nan_to_num(chex_ds.labels,0)
Chexpert_datamodule = torch.utils.data.DataLoader(chex_ds,batch_size=500,pin_memory=False)


def get_cavs(dataset, target, limit=5): 
    config.model.model._target_ = "torchxrayvision.autoencoders.ResNetAE"
    config.model.model.weights = "101-elastic"
    config.model.concept_storage = "label"
    config.model.concept = dataset.pathologies.index(target)
    model: LightningModule = hydra.utils.instantiate(config.model)
    model = model.to(device=0)

    sampler=ImbalancedDatasetSampler(dataset,callback_get_label=lambda x: x.labels[:,dataset.pathologies.index(target)])
    loader = torch.utils.data.DataLoader(dataset,batch_size=500,sampler=sampler,pin_memory=False)

    cavs = []
    for i,batch in enumerate(loader):
        batch['img'] = batch['img'].to(device=0)
        emb_pos, emb_neg, y_pos, y_neg = model.step(batch)
        print(len(emb_pos)/len(emb_neg))
        cavs.append(model.calculate_vectors(emb_pos, emb_neg))
        if i == limit:
            break
    return cavs


# %%
def eval(ae, generate_vector, target, data,method=None):
    if method:   
        wandb.init(project="concept_vector_stability",name=str(method)+" "+target)
    else:
        wandb.init(project="concept_vector_stability")
    result = []
    per_sample_table = None
    for sample in data.data_train:
        if sample["label"][data.pathologies.index(target)] == 1: 
            if target not in sample["pathology_masks"].keys():
                #print("no mask found")
                continue
            image = torch.from_numpy(sample["img"]).unsqueeze(0).cuda()
            vector = generate_vector(image,target)
            dimage = generate_explanation(sample, vector, target, ae=ae, clf=clf)
            metrics = calc_iou(dimage, sample["pathology_masks"][target]["mask"])
            recon = ae(image)["out"]
            metrics["mse"] = float(((image-recon)**2).mean().detach().cpu().numpy())
            metrics["mae"] = float(torch.abs(image-recon).mean().detach().cpu().numpy())
            metrics["idx"] = sample["__key__"]
            metrics["method"] = method
            #metrics["p"] = float(p)
            metrics["target"] = target
            if per_sample_table: 
                per_sample_table.add_data(*list(metrics.values()))
            else: 
                per_sample_table = wandb.Table(dataframe=pd.DataFrame(metrics,index=[metrics['idx']]))
            result.append(metrics)
            image = wandb.Image(image, caption="original image", masks= {"predictions":{"mask_data":sample["pathology_masks"][target]["mask"][0]}})
            r_image = wandb.Image(recon,caption="reconstruction")
            mask = wandb.Image(sample["pathology_masks"][target]["mask"][0],caption="gt mask")
            fig = plt.imshow(dimage)
            wandb.log({"mask":mask,"salincy": fig,"original":image,"reconstruction":r_image,"target":target,"vector":vector})
    wandb.log({"res_table":per_sample_table,"total_table":pd.DataFrame(result).groupby("method").agg("mean")})
    return pd.DataFrame(result)

# %%
for_eval = [
#            "Cardiomegaly",
           # 'Mass',
         #   'Nodule', 
            "Atelectasis",
            "Effusion",
         #   "Lung Opacity",
            ]

# %%
for target in for_eval:
    print(f"Starting exp for {target}")
    def calculate_perceptual_clusters(dataloader,target):
        vectors = []
        samples = []
        for sample in dataloader:
            if sample["label"][clf.pathologies.index(target)] == 1: 
                image = torch.tensor(sample['img']).clone().unsqueeze(0)
                image.requires_grad = True
                image_shape = image.shape
                image = image.to("cuda")
                vectors.append(generate_vector(image,target,ae=ae,clf=clf).to("cpu"))
                samples.append(sample)
        vectors = torch.stack(vectors)
        return vectors 
    vectors_chex = calculate_perceptual_clusters(chex_ds,target)
    vectors_nih = calculate_perceptual_clusters(nih_ds,target)
    print(f"Getting CAVs {target}")

    chexpert_train_cavs = get_cavs(chex_ds,target)
    nih_train_cavs = get_cavs(nih_ds,target)

    centroiders_nih = {10: ClusteredVectorFinder(10,vectors_nih), 20: ClusteredVectorFinder(20,vectors_nih), 
    50: ClusteredVectorFinder(50,vectors_nih),
    100: ClusteredVectorFinder(100,vectors_nih)}

    centroiders_chex = {10: ClusteredVectorFinder(10,vectors_chex), 20: ClusteredVectorFinder(20,vectors_chex), 
    50: ClusteredVectorFinder(50,vectors_chex),
    100: ClusteredVectorFinder(100,vectors_chex)}

    experiments = [
        {"method": "nih_cav_1 max", "function":lambda x,y : generate_vector_cav(nih_train_cavs,1,)},
        {"method": "nih_cav_2 max", "function":lambda x,y : generate_vector_cav(nih_train_cavs,2)},
        {"method": "nih_cav_mean max", "function":lambda x,y : generate_vector_cav(nih_train_cavs,1,mean=True)},
        {"method": "chex_cav_1 max", "function":lambda x,y : generate_vector_cav(chexpert_train_cavs,1)},
        {"method": "chex_cav_2 max", "function":lambda x,y : generate_vector_cav(chexpert_train_cavs,2)},
        {"method": "chex_cav_mean max", "function":lambda x,y : generate_vector_cav(chexpert_train_cavs,1,mean=True)},
        {"method": "latentshift max", "function": lambda x,y : generate_vector(x,y,ae=ae,clf=clf)},

        {"method": "centroid 10 all", "function": lambda x,y: centroiders_nih[10].find_closest(generate_vector(x,y,ae=ae,clf=clf))},
        {"method": "centroid 20 all", "function": lambda x,y: centroiders_nih[20].find_closest(generate_vector(x,y,ae=ae,clf=clf))},
        {"method": "centroid 50 all", "function": lambda x,y: centroiders_nih[50].find_closest(generate_vector(x,y,ae=ae,clf=clf))},
        {"method": "centroid 100 all", "function": lambda x,y: centroiders_nih[100].find_closest(generate_vector(x,y,ae=ae,clf=clf))},

        {"method": "centroid 10 all", "function": lambda x,y: centroiders_chex[10].find_closest(generate_vector(x,y,ae=ae,clf=clf))},
        {"method": "centroid 20 all", "function": lambda x,y: centroiders_chex[20].find_closest(generate_vector(x,y,ae=ae,clf=clf))},
        {"method": "centroid 50 all", "function": lambda x,y: centroiders_chex[50].find_closest(generate_vector(x,y,ae=ae,clf=clf))},
        {"method": "centroid 100 all", "function": lambda x,y: centroiders_chex[100].find_closest(generate_vector(x,y,ae=ae,clf=clf))}]

    for cfg in experiments:
        res = eval(ae,cfg['function'],target,NIH_datamodule,method=cfg['method'])

# %%



