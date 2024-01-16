import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ocpmodels.models.gemnet.layers.base_layers import DenseLayer,ResidualLayer



class Electronic_embedding(nn.Module):
    """Scale the embedding of charge and spin into the initial atomic embedding.
    """
    def __init__(
        self,
        num_features: int,
        num_residual: int = 2,
        activation: str = "swish",
    ) -> None:
        super(Electronic_embedding, self).__init__()
        self.linear_q = DenseLayer(num_features, num_features)
        self.linear_k = DenseLayer(1, num_features, bias=True)
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

        k = self.linear_k(e)[batch_seg]  
        embed=nn.functional.softplus(k*q)
        embed_total=torch.sum(embed,dim=1)
        pred_embed=embed_total.new_zeros(num_batch).index_add_(0, batch_seg, embed_total)
        per_natom=torch.bincount(batch_seg)
        per_natom=per_natom.to(torch.float32)
        embed_diff=E-pred_embed
        d_embed=torch.gather(embed_diff/per_natom,0,batch_seg)
        scale_embed=embed+d_embed.unsqueeze(-1).expand(embed.shape[0],embed.shape[1])/embed.shape[1]
        return self.resblock(scale_embed)