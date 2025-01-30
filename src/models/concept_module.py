from typing import Any, List
import inspect

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics.regression import CosineSimilarity

import numpy as np
from sklearn.linear_model import LogisticRegression
from src.models.components.metrics import UnitBallSampling
from einops import rearrange, reduce, repeat


class ConceptVectorModule(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, model: LightningModule, concept: str, concept_storage: str = "meta", target_class: int = 1):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.model = model
        self.model.requires_grad = False
        if inspect.ismethod(model.features): 
            self.model.features2 = torch.nn.Sequential(self.model.conv1,
                        self.model.bn1,
                        self.model.relu,
                        self.model.maxpool,
                        self.model.layer1,
                        self.model.layer2,
                        self.model.layer3,
                        self.model.layer4
                )
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # concept to find vectors for
        self.concept_storage = concept_storage
        self.concept = concept
        self.target_class = 0

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_sim_cav = UnitBallSampling()
        self.val_sim = UnitBallSampling()
        self.cav_val_sim = CosineSimilarity()

        self.test_sim = UnitBallSampling()

        # for logging best so far validation accuracy
        self.val_sim_best = MinMetric()

        # Different concept vectors
        self.cav = torch.ones((1024)).flatten(start_dim=0).cpu()

    def calculate_vectors(self, emb_pos, emb_neg):
        cav_model = LogisticRegression()
        x_train = np.concatenate(
            [
                emb_pos.detach().cpu().flatten(start_dim=1).numpy(),
                emb_neg.detach().cpu().flatten(start_dim=1).numpy(),
            ]
        )
        concept = np.concatenate(
            [np.ones(emb_pos.shape[0]), np.zeros(emb_neg.shape[0])]
        )
        cav_model.fit(x_train, concept)
        if len(cav_model.coef_) == 1:
            cav = torch.tensor(-cav_model.coef_[0])
        else:
            cav = -torch.tensor(cav_model.coef_)
        return cav

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def compute_grad(self, sample):
        self.model.classifier.zero_grad()
        with torch.enable_grad():
            prediction = self.model.classifier(sample)
            grad = torch.autograd.grad(
                prediction[self.target_class], list(self.model.classifier.parameters())
            )
        return grad[0][
            self.target_class
        ]  # We only need weight gradient, not bias gradient

    def get_gradient(self, data):
        sample_grads = torch.stack(list(map(self.compute_grad, data)))
        return sample_grads

    def step(self, batch: Any):
        img = batch["img"]
        if isinstance(self.concept,str):
            concept = batch[self.concept_storage][self.concept]
        else:
            concept = batch[self.concept_storage][:,self.concept].bool()

        y_pos = batch["label"][concept]
        y_neg = batch["label"][~concept]
        with torch.no_grad():
            emb_pos = self.model.features2(img[concept])
            emb_neg = self.model.features2(img[~concept])
       # pos_grad = self.get_gradient(emb_pos)
       # neg_grad = self.get_gradient(emb_neg)
        return emb_pos, emb_neg, y_pos, y_neg

    def training_step(self, batch: Any, batch_idx: int):
        emb_pos, emb_neg, y_pos, y_neg = self.step(batch)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()`
        # or else backpropagation will fail!
        return {"loss": torch.tensor(0), "emb_pos": emb_pos, "emb_neg": emb_neg}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        new_cav = torch.mean(torch.stack(list(map(lambda x: x["cav"], outputs))), dim=0)
        
        self.log(
            "cav",
            torch.nn.functional.mse_loss(new_cav, self.cav),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.cav = new_cav
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        emb_pos, emb_neg, y_pos, y_neg, pos_grad, neg_grad = self.step(batch)
        cav = self.calculate_vectors(emb_pos, emb_neg)

        # log val metrics
        cosine_cav = self.val_sim(pos_grad, self.cav)
        cav_val_sim = self.cav_val_sim(self.cav, cav)
        self.log(
            "val/cosine_cav", cosine_cav, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "val/cav_val_sim", cav_val_sim, on_step=True, on_epoch=True, prog_bar=False
        )

        return {"cosine_cav": cosine_cav}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_sim_cav.reset()
        self.train_sim_cav.reset()
        self.val_sim.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters())
