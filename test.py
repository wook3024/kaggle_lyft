# from __future__ import print_function, division, absolute_import
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

from efficientnet_pytorch import  EfficientNet

import warnings
warnings.filterwarnings("ignore")


l5kit.__version__
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
set_seed(42)

# ## Configs


# --- Lyft configs ---
cfg = {
    'format_version': 4,
    'data_path': "../input/lyft-motion-prediction-autonomous-vehicles",
    'model_params': {
        'model_architecture': 'resnet18',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "resnet18+267+adamp+0.2+10+1e-4+separate_modes_hard+threshold_0.5",
        # 'model_name': "lr_finder_test",
        'lr': 1e-4,
        # 'weight_path': "./result/test/resnet18+267+adamp+0.5+10_history_num_frames/models/save_model_1.pth",
        # 'weight_path': "./model_resnet34_output_0.pth",
        'weight_path': None,
        'train': True,
        'predict': False
    },

    'raster_params': {
        'raster_size': [267, 267],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5,
        'disable_traffic_light_faces': False
        
    },

    'train_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 8
    },
    
    'test_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4
    },
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4
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

val_cfg = cfg["val_data_loader"]
rasterizer = build_rasterizer(cfg, dm)
val_zarr = ChunkedDataset(dm.require(val_cfg["key"])).open()
val_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# cfg["raster_params"]["map_type"] = "py_satellite"
# satellite_rasterizer = build_rasterizer(cfg, dm)
# satellite_dataset = AgentDataset(cfg, train_zarr, rasterizer)

# print(train_dataset)
# print(len(train_daa))
# for i in range(len(train_dataset)):
#     # train_image_tensor = torch.from_numpy(train['image'])
#     # satellite_image_tensor = torch.from_numpy(satellite['image'])
#     train_dataset[i]['image'] = np.concatenate((train_dataset[i]['image'], satellite_dataset[i]['image']), axis=0)
#     print(train_dataset[i]['image'].shape)
#     # plt.figure(figsize = (8,6))
#     # im = np.array(train["image"]).transpose(1, 2, 0)
#     # im = train_dataset.rasterizer.to_rgb(im)
#     # target_positions_pixels = transform_points(train["target_positions"] + train["centroid"][:2], train["world_to_image"])
#     # draw_trajectory(np.array(im), target_positions_pixels, TARGET_POINTS_COLOR, radius=1, yaws=train["target_yaws"])

#     # plt.title(title)
#     # plt.imshow(im[::-1])
#     # plt.show()
#     if i > 10:
#         break

# # for i, train in enumerate(train_dataset):
# #     print(train['image'].shape)
# #     if i > 10:
# #         break
# assert 1 > 2

# train_dataset['image'] = torch.cat((train_dataset['image'], satellite_dataset['image']), 2)
# print(train_dataset)
# assert 1 > 2


train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"])
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"], 
                             num_workers=val_cfg["num_workers"])                             
print("==================================TRAIN DATA==================================")
print(train_dataset)
print("==================================VAL DATA==================================")
print(val_dataset)


# ## Simple visualization

# Let us visualize how an input to the model looks like.


def visualize_trajectory(dataset, index, title="target_positions movement with draw_trajectory"):
    data = dataset[index]
    im = data["image"].transpose(1, 2, 0)
    im = dataset.rasterizer.to_rgb(im)
    target_positions_pixels = transform_points(data["target_positions"] + data["centroid"][:2], data["world_to_image"])
    draw_trajectory(im, target_positions_pixels, TARGET_POINTS_COLOR, radius=1, yaws=data["target_yaws"])

    plt.title(title)
    plt.imshow(im[::-1])
    plt.show()


# for i in range(3):
#     plt.figure(figsize = (8,6))
#     visualize_trajectory(train_dataset, index=(90 + i))

# ## Loss function

# For this competition it is important to use the correct loss function when train the model. Our goal is to predict three possible paths together with the confidence score, so we need to use the loss function that takes that into account, simply using RMSE will not lead to an accurate model. More information about the loss function can be found here [negative log likelihood](https://github.com/lyft/l5kit/blob/master/competition.md).


# --- Function utils ---
# Original code from https://github.com/lyft/l5kit/blob/20ab033c01610d711c3d36e1963ecec86e8b85b6/l5kit/l5kit/evaluation/metrics.py
import numpy as np

import torch
from torch import Tensor


def pytorch_neg_multi_log_likelihood_batch(
    gt: Tensor, pred: Tensor, confidences: Tensor, avails: Tensor
) -> Tensor:
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    log-sum-exp trick is used here to avoid underflow and overflow, For more information about it see:
    https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    https://leimao.github.io/blog/LogSumExp/
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    assert len(pred.shape) == 4, f"expected 3D (MxTxC) array for pred, got {pred.shape}"
    batch_size, num_modes, future_len, num_coords = pred.shape

    assert gt.shape == (batch_size, future_len, num_coords), f"expected 2D (Time x Coords) array for gt, got {gt.shape}"
    assert confidences.shape == (batch_size, num_modes), f"expected 1D (Modes) array for gt, got {confidences.shape}"
    assert torch.allclose(torch.sum(confidences, dim=1), confidences.new_ones((batch_size,))), f"confidences should sum to 1, got \n{confidences} \n{torch.sum(confidences, dim=1)} \n{confidences.new_ones((batch_size,))}"
    assert avails.shape == (batch_size, future_len), f"expected 1D (Time) array for gt, got {avails.shape}"
    # assert all data are valid
    assert torch.isfinite(pred).all(), "invalid value found in pred"
    assert torch.isfinite(gt).all(), "invalid value found in gt"
    assert torch.isfinite(confidences).all(), "invalid value found in confidences"
    assert torch.isfinite(avails).all(), "invalid value found in avails"

    # convert to (batch_size, num_modes, future_len, num_coords)
    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords

    # error (batch_size, num_modes, future_len)
    error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

    with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = torch.log(confidences + 1e-10) - 0.5 * torch.sum(error, dim=-1)  # reduce time

    # use max aggregator on modes for numerical stability
    # error (batch_size, num_modes)
    max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
    error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
    # print("error", error)
    return torch.mean(error)


def pytorch_neg_multi_log_likelihood_single(
    gt: Tensor, pred: Tensor, avails: Tensor
) -> Tensor:
    """

    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        pred (Tensor): array of shape (bs)x(time)x(2D coords)
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """
    # pred (bs)x(time)x(2D coords) --> (bs)x(mode=1)x(time)x(2D coords)
    # create confidence (bs)x(mode=1)
    batch_size, future_len, num_coords = pred.shape
    confidences = pred.new_ones((batch_size, 1))
    return pytorch_neg_multi_log_likelihood_batch(gt, pred.unsqueeze(1), confidences, avails)

# ## Model

# Next we define the baseline model. Note that this model will return three possible trajectories together with confidence score for each trajectory.






class ResNet(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, dropout=0.0):
        super(ResNet, self).__init__()
        assert kernel_size % 2 == 1
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                out_features,
                out_features,
                kernel_size,
                stride=1,
                padding=(kernel_size - 1) // 2,
            ),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=201):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 1::2] = torch.cos(position * div_term)[:, : d_model // 2]
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class LyftMultiModel(LightningModule):

    def __init__(self):
        super().__init__()
        
        self.lr = cfg["model_params"]["lr"]
        self.num_modes = 3
        self.cur_epoch = 1
        self.start = time.time()
        self.iterations = []
        self.avg_losses = []
        self.avg_val_losses = []
        self.times = []
        self.avg_loss = 0.0
        model_name = cfg["model_params"]["model_name"]

        # self.save_hyperparameters(
        #     dict(
        #         # NUM_MODES = self.num_modes,
        #         # MODEL_NAME = model_name,
        #         # TRAIN_ZARR = cfg["train_data_loader"]["key"],
        #         # batch_size = cfg["train_data_loader"]["batch_size"],
        #         lr=cfg["model_params"]["lr"],
        #         # num_workers=cfg["train_data_loader"]["num_workers"]
        #     )
        # )

        try:
            os.mkdir(f'./result')
        except:
            pass
        try:
            os.mkdir(f'./result/test')
        except:
            pass
        try:
            os.mkdir(f'./result/test/{model_name}')
        except:
            pass
        try:
            os.mkdir(f'./result/test/{model_name}/results')
        except:
            pass
        try:
            os.mkdir(f'./result/test/{model_name}/models')
        except:
            pass
        try:
            os.mkdir(f'./result/test/{model_name}/logs')
        except:
            print(f'{model_name} folder is already exgist!')


        architecture = cfg["model_params"]["model_architecture"]
        backbone = eval(architecture)(pretrained=True, progress=True)
        # backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
        # backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        # backbone = torch.hub.load('narumiruna/efficientnet-pytorch', 'efficientnet_b0', pretrained=True)
        
        # print(backbone)
        

        if architecture == 'efficientnet':
            backbone.features = backbone.features[:17]
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        
        # num_in_channels *= 2


        if architecture == "resnest50" or architecture == "resnest18" or architecture == "resnest101":
            self.backbone.conv1[0] = nn.Conv2d(
                num_in_channels,
                self.backbone.conv1[0].out_channels,
                kernel_size=self.backbone.conv1[0].kernel_size,
                stride=self.backbone.conv1[0].stride,
                padding=self.backbone.conv1[0].padding,
                bias=False,
            )
        elif architecture == 'efficientnet':
            self.backbone.features[0][1] = nn.Conv2d(
                num_in_channels,
                self.backbone.features[0][1].out_channels,
                kernel_size=self.backbone.features[0][1].kernel_size,
                stride=self.backbone.features[0][1].stride,
                padding=self.backbone.features[0][1].padding,
                bias=False,
            )
        else:
            self.backbone.conv1 = nn.Conv2d(
                num_in_channels,
                self.backbone.conv1.out_channels,
                kernel_size=self.backbone.conv1.kernel_size,
                stride=self.backbone.conv1.stride,
                padding=self.backbone.conv1.padding,
                bias=False,
            )
        
        

        # This is 512 for resnet18 and resnet34;
        # And it is 2048 for the other resnets
        
        if architecture == "resnet50" or architecture == "resnext50" or architecture == "resnest50" or architecture == "resnest18" or architecture == "resnext101":
            backbone_out_features = 2048
        elif architecture == 'efficientnet':
            backbone_out_features = 320
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len


        # backbone_out_features = 256 + 128 + 64 + 32
        out_features = 256 + 128 + 64 + 32 + 16
        # You can add more layers here.
        # self.head = nn.Sequential(
        #     # nn.Dropout(0.2),
        #     nn.Linear(in_features=out_features, out_features=4096),
        # )

        self.num_preds = num_targets * self.num_modes

        # self.logit = nn.Linear(4096, out_features=self.num_preds + self.num_modes)
        self.x_preds = nn.Linear(4096, out_features=self.num_preds)
        self.x_modes = nn.Linear(4096, out_features=self.num_modes)

        self.pool2d = nn.AdaptiveAvgPool2d((1,1))


        dropout_rate = 0.2

        self.rnn1 = nn.LSTM(
            512,
            512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.deconv1 = nn.ConvTranspose1d(
            1024, 512, kernel_size=3, stride=2, padding=1
        )
        self.rnn2 = nn.LSTM(
            1024,
            512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.deconv2 = nn.ConvTranspose1d(
            1024, 512, kernel_size=3, stride=2, padding=1
        )
        out_features = 1024 
        
        self.pool = nn.AdaptiveAvgPool2d((3,3))
        self.pool2 = nn.AdaptiveAvgPool2d((32,64))
        self.pool3d = nn.AdaptiveAvgPool3d((1,512,1))

        self.head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=2048, out_features=4096),
        )

        self.linear = nn.Linear(512, 2048)
        self.scale = math.sqrt(2048)
        self.pe = PositionalEncoding(2048, dropout_rate)
        encoder_layer1 = nn.TransformerEncoderLayer(
            2048,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout_rate,
            activation="gelu",
        )
        self.transformer1 = nn.TransformerEncoder(encoder_layer1, 1)
        self.deconv1 = nn.ConvTranspose1d(
            2048, 1024, kernel_size=3, stride=2, padding=1
        )
        encoder_layer2 = nn.TransformerEncoderLayer(
            1024,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout_rate,
            activation="gelu",
        )
        self.transformer2 = nn.TransformerEncoder(encoder_layer2, 1)
        self.deconv2 = nn.ConvTranspose1d(
            2048, 1024, kernel_size=3, stride=2, padding=1
        )
        out_features = 2048

        self.rnn = nn.LSTM(
            out_features,
            512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        out_features = 1024

        # self.resnet1 = ResNet(
        #     num_in_channels, 512, kernel_size=3, dropout=dropout_rate
        # )
        # self.deconv1 = nn.ConvTranspose2d(
        #     512, 512, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet2 = ResNet(
        #     num_in_channels, 256, kernel_size=5, dropout=dropout_rate
        # )
        # self.deconv2 = nn.ConvTranspose2d(
        #     256, 256, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet3 = ResNet(
        #     256, 128, kernel_size=7, dropout=dropout_rate
        # )
        # self.deconv3 = nn.ConvTranspose2d(
        #     128, 128, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet4 = ResNet(
        #     128, 64, kernel_size=9, dropout=dropout_rate
        # )
        # self.deconv4 = nn.ConvTranspose2d(
        #     64, 64, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet5 = ResNet(
        #     64, 32, kernel_size=11, dropout=dropout_rate
        # )
        # self.deconv5 = nn.ConvTranspose2d(
        #     32, 32, kernel_size=3, stride=2, padding=1
        # )

        # self.resnet1 = ResNet(
        #     backbone_out_features, 256, kernel_size=3, dropout=dropout_rate
        # )
        # self.resnet2 = ResNet(
        #     backbone_out_features, 128, kernel_size=3, dropout=dropout_rate
        # )
        # self.resnet3 = ResNet(
        #     backbone_out_features, 64, kernel_size=3, dropout=dropout_rate
        # )
        # self.resnet4 = ResNet(
        #     backbone_out_features, 32, kernel_size=3, dropout=dropout_rate
        # )
        # self.resnet5 = ResNet(
        #     backbone_out_features, 16, kernel_size=3, dropout=dropout_rate
        # )
        
        

    def forward(self, x):


        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # x = self.backbone.features(x)
        
        # print("################", x.shape)
        # outs = []
        # # for i in range(2, 6):
        # #     # print("#########################", i)
        # #     x = getattr(self, f"resnet{i}")(x)
        # #     out = getattr(self, f"deconv{i}")(x)
        # #     outs.append(out)
        # for i in range(1, 6):
        #     out = getattr(self, f"resnet{i}")(x)
        #     outs.append(out)
        # x = torch.cat(outs, dim=1)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        # # print("################", x.shape)
        # x = self.pool3d(x)
        # x = torch.flatten(x, 2)

        # # print("################", x.shape)
        # outs = []
        # x, _ = self.rnn1(x)
        # out = self.deconv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        # outs.append(out)
        # x, _ = self.rnn2(x)
        # out = self.deconv2(x.permute(0, 2, 1)).permute(0, 2, 1)
        # outs.append(out)
        # x = torch.cat(outs, dim=-1)
        
        # # print("################", x.shape)
        

        # # print("################", x.shape)
        # x = self.pool2(x)
        # x = torch.flatten(x, 1)
        # # print("################", x.shape)
        # print("################", x.shape)
        # x = torch.relu(self.linear(x))
        # x = self.pe(x.permute(1, 0, 2))
        # outs = []
        # x = self.transformer1(x)
        # out = self.deconv1(x.permute(1, 2, 0)).permute(2, 0, 1)
        # outs.append(out)
        # x = self.transformer1(x)
        # out = self.deconv2(x.permute(1, 2, 0)).permute(2, 0, 1)
        # outs.append(out)
        # x = torch.cat(outs, dim=-1).permute(1, 0, 2)
        # # print("################", x.shape)
        # x = self.pool2(x)
        # x = torch.flatten(x, 1)
        # # print("################", x.shape)
        x = self.head(x)
        # x = self.logit(x)
        
        preds = self.x_preds(x)
        modes = self.x_modes(x)

        return preds, modes
    
    # def training_step(self, batch, batch_idx):
    #     inputs = batch["image"]
    #     target_availabilities = batch["target_availabilities"]
    #     targets = batch["target_positions"]
    #     x = self(inputs)
    #     bs, _ = x.shape
    #     pred, confidences = torch.split(x, self.num_preds, dim=1)
    #     preds = pred.view(bs, self.num_modes, self.future_len, 2)
    #     assert confidences.shape == (bs, self.num_modes)
    #     confidences = torch.softmax(confidences, dim=1)
    #     # Forward pass
    #     loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)
    #     # return pred, confidences
    #     return loss

    def training_step(self, batch, batch_idx):
        inputs = batch["image"]
        target_availabilities = batch["target_availabilities"]
        targets = batch["target_positions"]
        # x = self(inputs)
        pred, confidences = self(inputs)
        bs, _ = pred.shape
        # pred, confidences = torch.split(x, self.num_preds, dim=1)
        preds = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        # Forward pass
        loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)
        # return pred, confidences

        # train_logs = {
        #     'loss': loss,
        # }
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        inputs = batch["image"]
        target_availabilities = batch["target_availabilities"]
        targets = batch["target_positions"]
        # x = self(inputs)
        pred, confidences = self(inputs)
        bs, _ = pred.shape
        # pred, confidences = torch.split(x, self.num_preds, dim=1)
        preds = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        # Forward pass
        loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)
        # return pred, confidences

        val_logs = {
            'val_loss': loss,
        }
        return val_logs
    
    # def training_epoch_end(self, training_step_outputs):
    #     avg_loss = torch.mean(torch.tensor([x['loss'] for x in training_step_outputs]))
    #     self.avg_loss = avg_loss.item()
        

    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.mean(torch.tensor([x['val_loss'] for x in validation_step_outputs])).item()
        # avg_loss = self.avg_loss

        # print(f'val_loss: {avg_val_loss}, loss: {avg_loss}')
        print(f'val_loss: {avg_val_loss}')

        # tensorboard_logs = {'val_loss': avg_val_loss, "loss": avg_loss}
        tensorboard_logs = {'val_loss': avg_val_loss}

        torch.save(model.state_dict(), f'{os.getcwd()}/result/test/{model_name}/models/save_model_{self.cur_epoch}.pth')
        self.iterations.append(self.cur_epoch)
        # self.avg_losses.append(avg_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.times.append((time.time()-self.start)/60)
        self.start = time.time()
        
        results = pd.DataFrame({'iterations': self.iterations, 'avg_val_losses': self.avg_val_losses, 'elapsed_time (mins)': self.times})
        results.to_csv(f"{os.getcwd()}/result/test/{model_name}/results/train_metric{self.cur_epoch}.csv", index = False)

        return {
            'val_loss': avg_val_loss,
            # 'loss': avg_loss,
            'log': tensorboard_logs,
            # "progress_bar": {"val_loss": tensorboard_logs["val_loss"], "loss": tensorboard_logs["loss"]}
        }

    

    def configure_optimizers(self):
        optimizer = AdamP(model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)
        # optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max=self.epochs,
        #     eta_min=1e-5,
        # )
        return optimizer
    
    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader


# def forward(data, model, device, criterion = pytorch_neg_multi_log_likelihood_batch):
#     inputs = data["image"].to(device)
#     target_availabilities = data["target_availabilities"].to(device)
#     targets = data["target_positions"].to(device)
#     # Forward pass
#     preds, confidences = model(inputs)
#     loss = criterion(targets, preds, confidences, target_availabilities)
#     return loss, preds, confidences

model_name = cfg["model_params"]["model_name"]
checkpoint_callback = ModelCheckpoint(
    filepath=Path(f'{os.getcwd()}/result/test/{model_name}/models'),
    save_top_k=-1,
    verbose=True,
    prefix='pytorch_lightning'
)

logger = TensorBoardLogger(
    save_dir=Path(f'{os.getcwd()}/result/test/{model_name}/logs'),
    version=1,
    name='lightning_logs'
)

model = LyftMultiModel()

weight_path = cfg["model_params"]["weight_path"]
if weight_path:
    model.load_state_dict(torch.load(weight_path))

trainer = Trainer(max_epochs=5, logger=logger, checkpoint_callback=checkpoint_callback, limit_val_batches=0.2, gpus=[0])

# # Run learning rate finder
# lr_finder = trainer.tuner.lr_find(model)

# # Results can be found in
# lr_finder.results

# # Plot with
# fig = lr_finder.plot(suggest=True)
# fig.show()

# # Pick point based on plot, or get suggestion
# new_lr = lr_finder.suggestion()
# print(f'lr_finder: {lr_finder}')
# print(f'new_lr: {new_lr}')

# # update hparams of the model
# model.lr = new_lr

# # Fit model
# trainer.fit(model)
trainer.fit(model)

# trainer = Trainer(auto_lr_find=True, gpus=[1])
# trainer.tune(model)













########################################################################################################################

"""
ResNet code gently borrowed from
https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""


pretrained_settings = {
    'se_resnext101_32x4d': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
            'input_space': 'RGB',
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }
    },
}


class SEModule(nn.Module):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class Bottleneck(nn.Module):
    """
    Base class for bottlenecks that implements `forward()` method.
    """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out


class SEBottleneck(Bottleneck):
    """
    Bottleneck for SENet154.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)
        self.conv2 = nn.Conv2d(planes * 2, planes * 4, kernel_size=3,
                               stride=stride, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 4)
        self.conv3 = nn.Conv2d(planes * 4, planes * 4, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNetBottleneck(Bottleneck):
    """
    ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
    implementation and uses `stride=stride` in `conv1` and not in `conv2`
    (the latter is used in the torchvision implementation of ResNet).
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None):
        super(SEResNetBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
                               stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1,
                               groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SEResNeXtBottleneck(Bottleneck):
    """
    ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
    """
    expansion = 4

    def __init__(self, inplanes, planes, groups, reduction, stride=1,
                 downsample=None, base_width=4):
        super(SEResNeXtBottleneck, self).__init__()
        width = math.floor(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
                               stride=1)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)
        self.downsample = downsample
        self.stride = stride


class SENet(nn.Module):

    def __init__(self, block, layers, groups, reduction, dropout_p=0.2,
                 inplanes=128, input_3x3=True, downsample_kernel_size=3,
                 downsample_padding=1, num_classes=1000):
        """
        Parameters
        ----------
        block (nn.Module): Bottleneck class.
            - For SENet154: SEBottleneck
            - For SE-ResNet models: SEResNetBottleneck
            - For SE-ResNeXt models:  SEResNeXtBottleneck
        layers (list of ints): Number of residual blocks for 4 layers of the
            network (layer1...layer4).
        groups (int): Number of groups for the 3x3 convolution in each
            bottleneck block.
            - For SENet154: 64
            - For SE-ResNet models: 1
            - For SE-ResNeXt models:  32
        reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
            - For all models: 16
        dropout_p (float or None): Drop probability for the Dropout layer.
            If `None` the Dropout layer is not used.
            - For SENet154: 0.2
            - For SE-ResNet models: None
            - For SE-ResNeXt models: None
        inplanes (int):  Number of input channels for layer1.
            - For SENet154: 128
            - For SE-ResNet models: 64
            - For SE-ResNeXt models: 64
        input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
            a single 7x7 convolution in layer0.
            - For SENet154: True
            - For SE-ResNet models: False
            - For SE-ResNeXt models: False
        downsample_kernel_size (int): Kernel size for downsampling convolutions
            in layer2, layer3 and layer4.
            - For SENet154: 3
            - For SE-ResNet models: 1
            - For SE-ResNeXt models: 1
        downsample_padding (int): Padding for downsampling convolutions in
            layer2, layer3 and layer4.
            - For SENet154: 1
            - For SE-ResNet models: 0
            - For SE-ResNeXt models: 0
        num_classes (int): Number of outputs in `last_linear` layer.
            - For all models: 1000
        """
        super(SENet, self).__init__()
        self.inplanes = inplanes
        if input_3x3:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, 64, 3, stride=2, padding=1,
                                    bias=False)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('conv2', nn.Conv2d(64, 64, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('conv3', nn.Conv2d(64, inplanes, 3, stride=1, padding=1,
                                    bias=False)),
                ('bn3', nn.BatchNorm2d(inplanes)),
                ('relu3', nn.ReLU(inplace=True)),
            ]
        else:
            layer0_modules = [
                ('conv1', nn.Conv2d(3, inplanes, kernel_size=7, stride=2,
                                    padding=3, bias=False)),
                ('bn1', nn.BatchNorm2d(inplanes)),
                ('relu1', nn.ReLU(inplace=True)),
            ]
        # To preserve compatibility with Caffe weights `ceil_mode=True`
        # is used instead of `padding=1`.
        layer0_modules.append(('pool', nn.MaxPool2d(3, stride=2,
                                                    ceil_mode=True)))
        self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        self.layer1 = self._make_layer(
            block,
            planes=64,
            blocks=layers[0],
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=1,
            downsample_padding=0
        )
        self.layer2 = self._make_layer(
            block,
            planes=128,
            blocks=layers[1],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer3 = self._make_layer(
            block,
            planes=256,
            blocks=layers[2],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.layer4 = self._make_layer(
            block,
            planes=512,
            blocks=layers[3],
            stride=2,
            groups=groups,
            reduction=reduction,
            downsample_kernel_size=downsample_kernel_size,
            downsample_padding=downsample_padding
        )
        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.dropout = nn.Dropout(dropout_p) if dropout_p is not None else None
        self.last_linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)

    def features(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def logits(self, x):
        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def initialize_pretrained_model(model, num_classes, settings):
    assert num_classes == settings['num_classes'], \
        'num_classes should be {}, but is {}'.format(
            settings['num_classes'], num_classes)
    model.load_state_dict(model_zoo.load_url(settings['url']))
    model.input_space = settings['input_space']
    model.input_size = settings['input_size']
    model.input_range = settings['input_range']
    model.mean = settings['mean']
    model.std = settings['std']

def se_resnext101(num_classes=1000, pretrained='imagenet'):
    model = SENet(SEResNeXtBottleneck, [3, 4, 23, 3], groups=32, reduction=16,
                  dropout_p=None, inplanes=64, input_3x3=False,
                  downsample_kernel_size=1, downsample_padding=0,
                  num_classes=num_classes)
    if pretrained is not None:
        settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
        initialize_pretrained_model(model, num_classes, settings)
    return model


# model = se_resnext101()

# model.last_linear = nn.Linear(model.last_linear.in_features, NCLASSES)

# model = model.cuda()

# criterion = nn.BCEWithLogitsLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0005)
# scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

###################################################################################################################################








