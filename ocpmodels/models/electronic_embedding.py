import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ocpmodels.models.gemnet.layers.base_layers import DenseLayer,ResidualLayer



class Electronic_embedding(nn.Module):
    """Electronic embedding module. Incorporate the embedding of charge and spin into the initial atomic embedding.
        refer to SpookyNet https://github.com/OUnke/SpookyNet
    """
    def __init__(
        self,
        num_features: int,
        num_residual: int = 2,
        activation: str = "swish",
    ) -> None:
        super(Electronic_embedding, self).__init__()
        self.linear_q = DenseLayer(num_features, num_features)
        self.linear_k = DenseLayer(1, num_features, bias=False)
        self.linear_v = DenseLayer(1, num_features, bias=False)
        self.resblock = ResidualLayer(units=num_features, 
        nLayers=num_residual, 
        layer=DenseLayer,
        activation = activation
        )
    def forward(
        self,
        x: torch.Tensor,
        E: torch.Tensor,
        num_batch: int,
        batch_seg: torch.Tensor,
        mask: Optional[torch.Tensor] = None,  # only for backwards compatibility
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Evaluate interaction block.
        N: Number of atoms.

        x (FloatTensor [N, num_features]):
            Atomic feature vectors.
        """
        if batch_seg is None:  # assume a single batch
            batch_seg = torch.zeros(x.size(0), dtype=torch.int64, device=x.device)
        q = self.linear_q(x)  # queries
        E=E.to(torch.float32)
        e = torch.abs(E).unsqueeze(-1) # +/- spin is the same => abs

        k = self.linear_k(e)[batch_seg]  # keys
        v = self.linear_v(e)[batch_seg]  # values
        
        dot = torch.sum(k * q, dim=-1) / k.shape[-1] ** 0.5  # scaled dot product   
        
        a = nn.functional.softplus(dot)  # unnormalized attention weights
        
        anorm = a.new_zeros(num_batch).index_add_(0, batch_seg, a)
        if a.device.type == "cpu":  # indexing is faster on CPUs
            anorm = anorm[batch_seg]
        else:  # gathering is faster on GPUs
            anorm = torch.gather(anorm, 0, batch_seg)
        return self.resblock((a / (anorm + eps)).unsqueeze(-1) * v)