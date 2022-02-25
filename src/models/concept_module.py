from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MinMetric
from torchmetrics.regression import CosineSimilarity

import numpy as np
from sklearn.linear_model import LogisticRegression


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

    def __init__(self, model: LightningModule, concept: str, target_class: int = 1):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        self.model = model
        self.model.requires_grad = False
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # concept to find vectors for
        self.concept = concept

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_sim_cav = CosineSimilarity()
        self.train_sim_diff = CosineSimilarity()
        self.val_sim = CosineSimilarity()
        self.test_sim = CosineSimilarity()

        # for logging best so far validation accuracy
        self.val_sim_best = MinMetric()

        # Different concept vectors
        self.cav = torch.ones((1024, 7, 7)).flatten(start_dim=0).cpu()
        self.diff = torch.ones_like(self.cav).cpu()
        self.grad_vec = torch.ones_like(self.cav).cpu()

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
        diff = (
            (torch.mean(emb_pos, dim=0) - torch.mean(emb_neg, dim=0))
            .flatten(start_dim=0)
            .detach()
            .cpu()
        )
        return cav, diff, 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        img = batch["img"]
        concept = batch["meta"][self.concept]
        y_pos = batch["label"][concept]
        y_neg = batch["label"][~concept]
        with torch.no_grad():
            emb_pos = self.model.features(img[concept])
            emb_neg = self.model.features(img[~concept])

        return emb_pos, emb_neg, y_pos, y_neg

    def training_step(self, batch: Any, batch_idx: int):
        emb_pos, emb_neg, y_pos, y_neg = self.step(batch)
        cav, diff, _ = self.calculate_vectors(emb_pos, emb_neg)
        # log train metrics
        cosine_cav = self.train_sim_cav(self.cav, cav)
        cosine_diff = self.train_sim_diff(self.diff, diff)

        self.log(
            "train/cosine_cav", cosine_cav, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "train/cosine_diff",
            cosine_diff,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()`
        # or else backpropagation will fail!
        return {"loss": torch.tensor(0), "cav": cav, "diff": diff}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        self.cav = torch.mean(
            torch.stack(list(map(lambda x: x["cav"], outputs))), dim=0
        )
        self.diff = torch.mean(
            torch.stack(list(map(lambda x: x["diff"], outputs))), dim=0
        )
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        emb_pos, emb_neg, y_pos, y_neg = self.step(batch)
        cav, diff, _ = self.calculate_vectors(emb_pos, emb_neg)

        # log val metrics
        cosine_cav = self.val_sim(self.cav, cav)
        cosine_diff = self.val_sim(self.diff, diff)

        self.log(
            "val/cosine_cav", cosine_cav, on_step=True, on_epoch=True, prog_bar=False
        )
        self.log(
            "val/cosine_diff", cosine_diff, on_step=True, on_epoch=True, prog_bar=False
        )

        return {"cosine_cav": cosine_cav, "cosine_diff": cosine_diff}

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
