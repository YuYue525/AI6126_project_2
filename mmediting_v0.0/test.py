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
from mmcv.runner import load_checkpoint

import mmcv
import os

# Initialize distributed training (only need to initialize once), comment it if
# have already run this part
os.environ['RANK'] = '0'
os.environ['WORLD_SIZE'] = '1'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '50297'
init_dist('pytorch', **cfg.dist_params)

from mmedit.apis import single_gpu_test
from mmedit.datasets import build_dataloader
from mmcv.parallel import MMDataParallel

cfg.gpus = 1

# Build a test dataloader and model
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
        persistent_workers=False)

# Build the model
model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg).cuda()

load_checkpoint(model, './work_dirs/srgan/iter_40000.pth', map_location='cuda')

model = MMDataParallel(model, device_ids=[0])

outputs = single_gpu_test(model, data_loader, save_image=True,
                          save_path='./test/srgan/results')

