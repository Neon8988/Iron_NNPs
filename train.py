import torch
import torch_geometric
import logging
from pathlib import Path
from tqdm import tqdm
import os
from ocpmodels import models
from ocpmodels.common import logger
from ocpmodels.common.utils import setup_logging, load_config
from ocpmodels.datasets import LmdbDataset
from ocpmodels.common.registry import registry
from ocpmodels.trainers import EnergyTrainer, ForcesTrainer
setup_logging()
from pyparsing import identbodychars
from torch import seed



torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
#conf=load_config(config_path)[0]
task={'dataset': 'single_point_lmdb',
 'description': 'Regressing to energies for DFT Iron structures',
 'type': 'regression',
 'metric': 'mae',
 'labels': ['potential energy'],
 }

model={'name': 'schnet_scale',
 'hidden_channels': 16,
 'num_filters': 16,
 'num_interactions': 4,
 'num_gaussians': 4,
 'cutoff': 6.0,
 'max_neighbors': 50,
 'use_pbc': False,
 'otf_graph': True,
 'regress_forces': False,
 'only_electron': True,
 'seperated': True,
 'readout': 'add',
'ewald_hyperparams': {'k_cutoff': 0.2,'delta_k':0.4,'num_k_rbf':4,'downprojection_size':2,'num_hidden':2}}

optimizer={'batch_size': 16,
 'eval_batch_size': 16,
 'num_workers': 4,
 'lr_initial': 0.0005,
 'lr_gamma': 0.1,
 'lr_milestones': [50000,100000, 150000],
 'warmup_steps': 50000,
 'warmup_factor': 0.2,
 'max_epochs': 10,}

name='schnet_scale'

logger='wandb'

dataset=[{'src': 'data/train.lmdb',
  'normalize_labels': True,
  'target_mean': -39.06976461742204,
  'target_std': 4.24765763408271} ,
 {'src': 'data/val.lmdb'},
 {'src': 'data/test.lmdb'}]

trainer=EnergyTrainer(task=task,
                      model=model,
                      dataset=dataset,
                      optimizer=optimizer,
                      identifier=name,
                      run_dir='runs',
                      is_debug=False,
                      print_every=1000,
                      seed=42,
                      logger=logger,
                      local_rank=0,
                      amp=False)
trainer.train()