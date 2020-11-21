from __future__ import print_function, division, absolute_import
from typing import Dict

from tempfile import gettempdir
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101
from tqdm import tqdm

from torch.nn.parallel.data_parallel import DataParallel

import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace
from l5kit.geometry import transform_points
from l5kit.visualization import PREDICTED_POINTS_COLOR, TARGET_POINTS_COLOR, draw_trajectory
from prettytable import PrettyTable
from pathlib import Path

import matplotlib.pyplot as plt

import os
import random
import time


from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo
from adamp import AdamP
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import gc


import warnings
warnings.filterwarnings("ignore")


l5kit.__version__


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)

# ## Configs

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "../../input/lyft-motion-prediction-autonomous-vehicles",
    'model_params': {
        'model_architecture': 'resnest50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        # 'model_name': "train_resnest50+267+adamp+0.5+15+1e-4+threshold_0.5+history_yaws+history_positions",
        'model_name': "test",
        'lr': 1e-4,
        'weight_path': "./result/train/train_resnest50+267+adamp+0.5+15+1e-4+threshold_0.5/pytorch_lightning-models-v51.ckpt",
        # 'weight_path': "./model_resnet34_output_0.pth",
        # 'weight_path': None,
        'train': True,
        'predict': False
    },

    'raster_params': {
        'raster_size': [0, 0],
        'pixel_size': [1.0, 1.0],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/train.zarr',
        'batch_size': 512,
        'shuffle': False,
        'num_workers': 24
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 8
    },
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 16
    },

    'train_params': {
        'max_num_steps': 201,
        'checkpoint_every_n_steps': 20,
    }
}

# Couple of things to note:
# 
#  - **model_architecture:** you can put 'resnet18', 'resnet34' or 'resnet50'. For the pretrained model we use resnet18 so we need to use 'resnet18' in the config.
#  - **weight_path:** path to the pretrained model. If you don't have a pretrained model and want to train from scratch, put **weight_path** = False. 
#  - **model_name:** the name of the model that will be saved as output, this is only when **train**= True.
#  - **train:** True if you want to continue to train the model. Unfortunately due to Kaggle memory constraint if **train**=True then you should put **predict** = False.
#  - **predict:** True if you want to predict and submit to Kaggle. Unfortunately due to Kaggle memory constraint if you want to predict then you need  to put **train** = False.
#  - **lr:** learning rate of the model, feel free to change as you see fit. In the future I also plan to implement learning rate decay. 
#  - **raster_size:** specify the size of the image, the default is [224,224]. Increase **raster_size** can improve the score. However the training time will be significantly longer. 
#  - **batch_size:** number of inputs for one forward pass, again one of the parameters to tune. 
#  - **max_num_steps:** the number of iterations to train, i.e. number of epochs.
#  - **checkpoint_every_n_steps:** the model will be saved at every n steps, again change this number as to how you want to keep track of the model.

# ## Load the train and test data


# set env variable for data
DIR_INPUT = cfg["data_path"]
os.environ["L5KIT_DATA_FOLDER"] = DIR_INPUT
dm = LocalDataManager(None)


# ===== INIT TRAIN DATASET============================================================
train_cfg = cfg["train_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
train_dataset = AgentDataset(cfg, train_zarr, rasterizer)

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"], pin_memory=True)

# val_cfg = cfg["val_data_loader"]
# rasterizer = build_rasterizer(cfg, dm)
# val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
# val_dataset = AgentDataset(cfg, train_zarr, rasterizer)

import pickle
import random
import time


start = time.time()


file_name = "target_dataset_soft5.pkl"
open_file = open(file_name, "wb")

create_dataset = []
tr_it = iter(train_dataloader)
for i in range(len(train_dataloader)):
    data = next(tr_it)
    target_positions = data["target_positions"].to("cuda")
    target_yaws = data["target_yaws"].to("cuda")
    target_availabilities = torch.tensor(data["target_availabilities"]).to("cuda")
    for idx in range(target_yaws.shape[0]):
        ori_position_x = target_positions[idx,:,0] * target_availabilities[idx]
        sub_position_x = torch.cat((target_positions[idx,:1,0], target_positions[idx,:-1,0]), 0) * target_availabilities[idx]
        val_num = torch.sum(target_availabilities[idx])
        internal_max = torch.max(torch.abs(ori_position_x - sub_position_x)) 
        interval_mean = torch.div(torch.sum(torch.abs(ori_position_x - sub_position_x)[1:]), val_num-1)
        # random 값 조절 가능!!!
        if (internal_max - interval_mean > 0.4 and interval_mean > 0.2) or torch.max(torch.abs(target_positions[idx,:,1])) > 2.0 or (torch.max(torch.abs(target_yaws[idx])) > 0.2 and interval_mean > 0.2)  or torch.max(torch.abs(ori_position_x)) > 90 or random.randint(1, 5) == 1:
            create_dataset.append(i*cfg["train_data_loader"]["batch_size"]+idx)
            # print(create_dataset[-1], len(create_dataset))
            # print(data["history_yaws"][idx])
            # print(train_dataset[i*cfg["train_data_loader"]["batch_size"]+idx]["history_yaws"])
    if i % 200 == 0:
        # print(batch["history_yaws"].shape)
        print("cur_step: ", (i+1))
        print("cur_datapoint: ", (i+1)*cfg["train_data_loader"]["batch_size"])
        print("size: ", len(create_dataset))
        print(f"time : {(time.time()-start)/60} mins")
        print("")
        pickle.dump(create_dataset, open_file)
    

save_file_name = "success_target_dataset_soft5.pkl"
save_open_file = open(save_file_name, "wb")

pickle.dump(create_dataset, open_file)
pickle.dump(create_dataset, save_open_file)
open_file.close()
save_open_file.close()
print(len(create_dataset))

# file_name = "dataset_sampling_soft5.pkl"
# open_file = open(file_name, "wb")

# create_dataset = []
# tr_it = iter(train_dataloader)
# for i in range(len(train_dataloader)):
#     data = next(tr_it)
#     history_positions = data["history_positions"].to("cuda")
#     history_yaws = data["history_yaws"].to("cuda")
#     for idx in range(data["history_yaws"].shape[0]):
#         if torch.max(torch.abs(history_positions[idx,:,1])) > 0.3 or torch.max(torch.abs(history_positions[idx,:,0])) > 13 or torch.max(torch.abs(history_yaws[idx])) > 0.05 or random.randint(1,5) == 1:
#             create_dataset.append(i*cfg["train_data_loader"]["batch_size"]+idx)
#             # print(create_dataset[-1], len(create_dataset))
#             # print(data["history_yaws"][idx])
#             # print(train_dataset[i*cfg["train_data_loader"]["batch_size"]+idx]["history_yaws"])
#     if i % 100 == 0:
#         # print(batch["history_yaws"].shape)
#         print("cur_step: ", (i+1))
#         print("cur_datapoint: ", (i+1)*cfg["train_data_loader"]["batch_size"])
#         print("size: ", len(create_dataset))
#         print(f"time : {(time.time()-start)/60} mins")
#         print("")
#         pickle.dump(create_dataset, open_file)

    

# save_file_name = "success_target_dataset_soft5.pkl"
# save_open_file = open(save_file_name, "wb")

# pickle.dump(create_dataset, save_open_file)
# pickle.dump(create_dataset, open_file)
# open_file.close()
# print(len(create_dataset))

# open_file = open(file_name, "rb")
# create_dataset = pickle.load(open_file)
# open_file.close()

# print(loaded_list)

# from torch.utils.data import Dataset

# class CustomDataset(Dataset): 
#   def __init__(self, create_dataset):
#     self.create_dataset = create_dataset

#   # 총 데이터의 개수를 리턴
#   def __len__(self): 
#     return len(self.create_dataset)

#   # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
#   def __getitem__(self, idx): 
#     data = train_dataset[idx]
#     return data

# train_dataset = CustomDataset(create_dataset)

# 50000 - 20m/s
# 500000 - 27m/s