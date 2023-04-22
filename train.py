import torch, torchvision
import mmedit
from mmcv import Config

cfg = Config.fromfile('srgan.py')
# print(f'Config:\n{cfg.pretty_text}')

import os.path as osp

from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.apis import train_model
from mmcv.runner import init_dist

import mmcv
import os

# Initialize distributed training (only need to initialize once), comment it if
# have already run this part
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '50297'
init_dist('pytorch', **cfg.dist_params)

# Use smaller batch size for training
cfg.data.workers_per_gpu = 0
cfg.data.val_workers_per_gpu = 0

cfg.gpus = 1

# Build dataset
datasets = [build_dataset(cfg.data.train)]

# Build the model
model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

print(sum(p.numel() for p in model.parameters()))

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

# Meta information
meta = dict()
if cfg.get('exp_name', None) is None:
    cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
meta['exp_name'] = cfg.exp_name
meta['mmedit Version'] = mmedit.__version__
meta['seed'] = 0

# Train the model
train_model(model, datasets, cfg, distributed=True, validate=True, meta=meta)

