from torchmetrics import Metric
import torch
from einops import rearrange, repeat


class UnitBallSampling(Metric):
    def __init__(self, dist_sync_on_step=False) -> None:
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sign", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("value", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, grads: torch.Tensor, concepts: torch.Tensor):
        concepts = repeat(concepts, "x ->b x", b=grads.shape[0])
        assert concepts.shape == grads.shape
        grads = grads.to(concepts.device)
        self.cosine_sum = torch.nn.functional.cosine_similarity(
            grads, concepts
        ) / torch.sqrt(torch.log(torch.tensor(grads.shape[1])) / grads.shape[1])
        self.sign += torch.sign(torch.sum(self.cosine_sum)).long()
        self.value += torch.sum(torch.abs(self.cosine_sum)).long()
        self.count += grads.shape[0]

    def compute(self):
        return self.sign * self.value / self.count
