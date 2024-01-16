"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)


class BaseModel(nn.Module):
    def __init__(self, num_atoms=None, bond_feat_dim=None, num_targets=None):
        super(BaseModel, self).__init__()
        self.num_atoms = num_atoms
        self.bond_feat_dim = bond_feat_dim
        self.num_targets = num_targets
        electron_config = torch.tensor([
        #  Z 1s 2s 2p 3s 3p 4s  3d 4p 5s  4d 5p 6s  4f  5d 6p vs vp  vd  vf
        [  0, 0, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 0, 0,  0,  0], # n
        [  1, 1, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # H
        [  2, 2, 0, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # He
        [  3, 2, 1, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Li
        [  4, 2, 2, 0, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Be
        [  5, 2, 2, 1, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0], # B
        [  6, 2, 2, 2, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0], # C
        [  7, 2, 2, 3, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 3,  0,  0], # N
        [  8, 2, 2, 4, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0], # O
        [  9, 2, 2, 5, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 5,  0,  0], # F
        [ 10, 2, 2, 6, 0, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # Ne
        [ 11, 2, 2, 6, 1, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Na
        [ 12, 2, 2, 6, 2, 0, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Mg
        [ 13, 2, 2, 6, 2, 1, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 1,  0,  0], # Al
        [ 14, 2, 2, 6, 2, 2, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 2,  0,  0], # Si
        [ 15, 2, 2, 6, 2, 3, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 3,  0,  0], # P
        [ 16, 2, 2, 6, 2, 4, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 4,  0,  0], # S
        [ 17, 2, 2, 6, 2, 5, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 5,  0,  0], # Cl
        [ 18, 2, 2, 6, 2, 6, 0,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 6,  0,  0], # Ar
        [ 19, 2, 2, 6, 2, 6, 1,  0, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # K
        [ 20, 2, 2, 6, 2, 6, 2,  0, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Ca
        [ 21, 2, 2, 6, 2, 6, 2,  1, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  1,  0], # Sc
        [ 22, 2, 2, 6, 2, 6, 2,  2, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  2,  0], # Ti
        [ 23, 2, 2, 6, 2, 6, 2,  3, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  3,  0], # V
        [ 24, 2, 2, 6, 2, 6, 1,  5, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0,  5,  0], # Cr
        [ 25, 2, 2, 6, 2, 6, 2,  5, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  5,  0], # Mn
        [ 26, 2, 2, 6, 2, 6, 2,  6, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  6,  0], # Fe
        [ 27, 2, 2, 6, 2, 6, 2,  7, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  7,  0], # Co
        [ 28, 2, 2, 6, 2, 6, 2,  8, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0,  8,  0], # Ni
        [ 29, 2, 2, 6, 2, 6, 1, 10, 0, 0,  0, 0, 0,  0,  0, 0, 1, 0, 10,  0], # Cu
        [ 30, 2, 2, 6, 2, 6, 2, 10, 0, 0,  0, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Zn
        [ 31, 2, 2, 6, 2, 6, 2, 10, 1, 0,  0, 0, 0,  0,  0, 0, 2, 1, 10,  0], # Ga
        [ 32, 2, 2, 6, 2, 6, 2, 10, 2, 0,  0, 0, 0,  0,  0, 0, 2, 2, 10,  0], # Ge
        [ 33, 2, 2, 6, 2, 6, 2, 10, 3, 0,  0, 0, 0,  0,  0, 0, 2, 3, 10,  0], # As
        [ 34, 2, 2, 6, 2, 6, 2, 10, 4, 0,  0, 0, 0,  0,  0, 0, 2, 4, 10,  0], # Se
        [ 35, 2, 2, 6, 2, 6, 2, 10, 5, 0,  0, 0, 0,  0,  0, 0, 2, 5, 10,  0], # Br
        [ 36, 2, 2, 6, 2, 6, 2, 10, 6, 0,  0, 0, 0,  0,  0, 0, 2, 6, 10,  0], # Kr
        [ 37, 2, 2, 6, 2, 6, 2, 10, 6, 1,  0, 0, 0,  0,  0, 0, 1, 0,  0,  0], # Rb
        [ 38, 2, 2, 6, 2, 6, 2, 10, 6, 2,  0, 0, 0,  0,  0, 0, 2, 0,  0,  0], # Sr
        [ 39, 2, 2, 6, 2, 6, 2, 10, 6, 2,  1, 0, 0,  0,  0, 0, 2, 0,  1,  0], # Y
        [ 40, 2, 2, 6, 2, 6, 2, 10, 6, 2,  2, 0, 0,  0,  0, 0, 2, 0,  2,  0], # Zr
        [ 41, 2, 2, 6, 2, 6, 2, 10, 6, 1,  4, 0, 0,  0,  0, 0, 1, 0,  4,  0], # Nb
        [ 42, 2, 2, 6, 2, 6, 2, 10, 6, 1,  5, 0, 0,  0,  0, 0, 1, 0,  5,  0], # Mo
        [ 43, 2, 2, 6, 2, 6, 2, 10, 6, 2,  5, 0, 0,  0,  0, 0, 2, 0,  5,  0], # Tc
        [ 44, 2, 2, 6, 2, 6, 2, 10, 6, 1,  7, 0, 0,  0,  0, 0, 1, 0,  7,  0], # Ru
        [ 45, 2, 2, 6, 2, 6, 2, 10, 6, 1,  8, 0, 0,  0,  0, 0, 1, 0,  8,  0], # Rh
        [ 46, 2, 2, 6, 2, 6, 2, 10, 6, 0, 10, 0, 0,  0,  0, 0, 0, 0, 10,  0], # Pd
        [ 47, 2, 2, 6, 2, 6, 2, 10, 6, 1, 10, 0, 0,  0,  0, 0, 1, 0, 10,  0], # Ag
        [ 48, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 0, 0,  0,  0, 0, 2, 0, 10,  0], # Cd
        [ 49, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 1, 0,  0,  0, 0, 2, 1, 10,  0], # In
        [ 50, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 2, 0,  0,  0, 0, 2, 2, 10,  0], # Sn
        [ 51, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 3, 0,  0,  0, 0, 2, 3, 10,  0], # Sb
        [ 52, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 4, 0,  0,  0, 0, 2, 4, 10,  0], # Te
        [ 53, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 5, 0,  0,  0, 0, 2, 5, 10,  0], # I
        [ 54, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 0,  0,  0, 0, 2, 6, 10,  0], # Xe
        [ 55, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1,  0,  0, 0, 1, 0,  0,  0], # Cs
        [ 56, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  0, 0, 2, 0,  0,  0], # Ba
        [ 57, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  0,  1, 0, 2, 0,  1,  0], # La
        [ 58, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  1,  1, 0, 2, 0,  1,  1], # Ce
        [ 59, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  3,  0, 0, 2, 0,  0,  3], # Pr
        [ 60, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  4,  0, 0, 2, 0,  0,  4], # Nd
        [ 61, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  5,  0, 0, 2, 0,  0,  5], # Pm
        [ 62, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  6,  0, 0, 2, 0,  0,  6], # Sm
        [ 63, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  0, 0, 2, 0,  0,  7], # Eu
        [ 64, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  7,  1, 0, 2, 0,  1,  7], # Gd
        [ 65, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2,  9,  0, 0, 2, 0,  0,  9], # Tb
        [ 66, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 10,  0, 0, 2, 0,  0, 10], # Dy
        [ 67, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 11,  0, 0, 2, 0,  0, 11], # Ho
        [ 68, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 12,  0, 0, 2, 0,  0, 12], # Er
        [ 69, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 13,  0, 0, 2, 0,  0, 13], # Tm
        [ 70, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  0, 0, 2, 0,  0, 14], # Yb
        [ 71, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  1, 0, 2, 0,  1, 14], # Lu
        [ 72, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  2, 0, 2, 0,  2, 14], # Hf
        [ 73, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  3, 0, 2, 0,  3, 14], # Ta
        [ 74, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  4, 0, 2, 0,  4, 14], # W
        [ 75, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  5, 0, 2, 0,  5, 14], # Re
        [ 76, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  6, 0, 2, 0,  6, 14], # Os
        [ 77, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14,  7, 0, 2, 0,  7, 14], # Ir
        [ 78, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14,  9, 0, 1, 0,  9, 14], # Pt
        [ 79, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 1, 14, 10, 0, 1, 0, 10, 14], # Au
        [ 80, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 0, 2, 0, 10, 14], # Hg
        [ 81, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 1, 2, 1, 10, 14], # Tl
        [ 82, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 2, 2, 2, 10, 14], # Pb
        [ 83, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 3, 2, 3, 10, 14], # Bi
        [ 84, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 4, 2, 4, 10, 14], # Po
        [ 85, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 5, 2, 5, 10, 14], # At
        [ 86, 2, 2, 6, 2, 6, 2, 10, 6, 2, 10, 6, 2, 14, 10, 6, 2, 6, 10, 14]  # Rn
        ])
        
        self.electron_configs = electron_config / electron_config[-1,:]

    def forward(self, data):
        raise NotImplementedError

    def generate_graph(
        self,
        data,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
                edge_index, cell_offsets, neighbors = radius_graph_pbc(
                    data, cutoff, max_neighbors
                )

            out = get_pbc_distances(
                data.pos,
                edge_index,
                data.cell,
                cell_offsets,
                neighbors,
                return_offsets=True,
                return_distance_vec=True,
            )

            edge_index = out["edge_index"]
            edge_dist = out["distances"]
            cell_offset_distances = out["offsets"]
            distance_vec = out["distance_vec"]
        else:
            if otf_graph:
                edge_index = radius_graph(
                    data.pos,
                    r=cutoff,
                    batch=data.batch,
                    max_num_neighbors=max_neighbors,
                )

            j, i = edge_index
            distance_vec = data.pos[j] - data.pos[i]

            edge_dist = distance_vec.norm(dim=-1)
            cell_offsets = torch.zeros(
                edge_index.shape[1], 3, device=data.pos.device
            )
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )
    

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())









        
        