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
        'model_architecture': 'resnest50',
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,
        'model_name': "resnet50+267+adamp+0.2+10+1e-4+separate_modes_hard+threshold_0.5+3dpool_(yaw+poisition)_cnn_repaire_remove_reakyRelu",
        # 'model_name': "test...",
        # 'model_name': "lr_finder_train",
        'lr': 1e-4,
        # 'weight_path': "./result/train/train_resnest50+267+adamp+0.5+15+1e-4+threshold_0.5/models/save_model_epoch1_20000.pth",
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
        'filter_agents_threshold': 0.5
    },

    'train_data_loader': {
        'key': 'scenes/sample.zarr',
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 4
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
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

train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"], pin_memory=True)
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"], 
                             num_workers=val_cfg["num_workers"], pin_memory=True)                             
print("==================================TRAIN DATA==================================")
# print(train_dataset)
print("==================================VAL DATA==================================")
# print(val_dataset)


# for i in range(100):
#     print(f"#####################history yaws_{i}#####################")
#     print(train_dataset[i]["history_yaws"])


# #====== INIT TEST DATASET=============================================================
# test_cfg = cfg["test_data_loader"]
# rasterizer = build_rasterizer(cfg, dm)
# test_zarr = ChunkedDataset(dm.require(test_cfg["key"])).open()
# test_mask = np.load(f"{DIR_INPUT}/scenes/mask.npz")["arr_0"]
# test_dataset = AgentDataset(cfg, test_zarr, rasterizer, agents_mask=test_mask)
# test_dataloader = DataLoader(test_dataset,shuffle=test_cfg["shuffle"],batch_size=test_cfg["batch_size"],
#                              num_workers=test_cfg["num_workers"])
# print("==================================TEST DATA==================================")
# # print(test_dataset)
# print("######################testdataset info#########################", test_dataset[0].keys())


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


# for i in range(11):
# plt.figure(figsize = (8,6))
# visualize_trajectory(train_dataset, index=(90))

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
            nn.BatchNorm1d(out_features),
            # nn.LeakyReLU(inplace=True),
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
            nn.BatchNorm1d(out_features),
            # nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.conv1(x)
        out = self.conv2(x) + x
        return out


class LyftMultiModel(LightningModule):

    def __init__(self, device_version):
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
        self.step = 0
        self.my_device = torch.device(f'cuda:{device_version}')
        self.previous_postiion_x = torch.tensor(train_dataset[0]["history_positions"][0, 0], dtype=torch.float32, device=self.my_device)
        self.previous_postiion_y = torch.tensor(train_dataset[0]["history_positions"][0, 1], dtype=torch.float32, device=self.my_device)
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
        # backbone = eval(architecture)(pretrained=True, progress=True)
        # backbone = torch.hub.load('pytorch/vision:v0.6.0', 'resnext101_32x8d', pretrained=True)
        backbone = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        
        # print(backbone)
        self.backbone = backbone

        num_history_channels = (cfg["model_params"]["history_num_frames"] + 1) * 2
        num_in_channels = 3 + num_history_channels
        
        # num_in_channels *= 2


        if architecture == "resnest50" or architecture == "resnest101":
            self.backbone.conv1[0] = nn.Conv2d(
                num_in_channels,
                self.backbone.conv1[0].out_channels,
                kernel_size=self.backbone.conv1[0].kernel_size,
                stride=self.backbone.conv1[0].stride,
                padding=self.backbone.conv1[0].padding,
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
        
        if architecture == "resnet50" or architecture == "resnext50" or architecture == "resnest50" or architecture == "resnext101":
            backbone_out_features = 2048
        else:
            backbone_out_features = 512

        # X, Y coords for the future positions (output shape: batch_sizex50x2)
        self.future_len = cfg["model_params"]["future_num_frames"]
        num_targets = 2 * self.future_len

        # You can add more layers here.
        # self.head = nn.Sequential(
        #     # nn.Dropout(0.2),
        #     nn.Linear(in_features=backbone_out_features, out_features=4096),
        # )
        
        # self.history_set_head = nn.Sequential(
        #     nn.Linear(in_features = 33, out_features = 64),
        #     nn.Linear(in_features=64, out_features=64),
        # )

        # self.yaw_head = nn.Sequential(
        #     # nn.Dropout(0.2),
        #     nn.Linear(in_features=11, out_features=64),
        #     nn.Linear(in_features=64, out_features=128),
        #     nn.Linear(in_features=128, out_features=256)
        # )

        # self.position_head = nn.Sequential(
        #     # nn.Dropout(0.2),
        #     nn.Linear(in_features=22, out_features=64),
        #     nn.Linear(in_features=64, out_features=128),
        #     nn.Linear(in_features=128, out_features=256)
        # )

        # self.pool2d = nn.AdaptiveMaxPool2d((128, 8))
        # self.pool3d = nn.AdaptiveMaxPool3d((11, 128, 1))
        # self.reduction_layer = nn.Sequential(
        #     nn.Linear(in_features=backbone_out_features, out_features=1024),
        #     nn.Linear(in_features=1024, out_features=512)
        # )

        self.parend_head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=2048+44, out_features=4000),

        )

        self.batchNorm = nn.BatchNorm1d(99)
        # self.dropOut = nn.Dropout(0.2)

        # 256 + 128 + 64 + 32

        self.num_preds = num_targets * self.num_modes


        self.x_preds = nn.Linear(99*64, out_features=self.num_preds)
        self.x_modes = nn.Linear(99*64, out_features=self.num_modes)

        num_feature = 3
        
        #################################################################
        # self.yaws_rnn1 = nn.LSTM(
        #     1,
        #     4,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        # self.positions_rnn1 = nn.LSTM(
        #     2,
        #     4,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )
        self.rnn1 = nn.LSTM(
            80,
            32,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.deconv1 = nn.ConvTranspose1d(
            64, 32, kernel_size=3, stride=2, padding=1
        )
        self.rnn2 = nn.LSTM(
            64,
            32,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.deconv2 = nn.ConvTranspose1d(
            64, 32, kernel_size=3, stride=2, padding=1
        )
        # out_features = 8



        
        # self.resnet1 = ResNet(
        #     128+3, 256, kernel_size=5, dropout=0.2
        # )
        # self.deconv1 = nn.ConvTranspose1d(
        #     256, 256, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet2 = ResNet(
        #     256, 128, kernel_size=7, dropout=0.2
        # )
        # self.deconv2 = nn.ConvTranspose1d(
        #     128, 128, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet1 = ResNet(
        #     128+3, 64, kernel_size=9, dropout=0.2
        # )
        # self.deconv1 = nn.ConvTranspose1d(
        #     64, 64, kernel_size=3, stride=2, padding=1
        # )
        # self.resnet2 = ResNet(
        #     64, 32, kernel_size=11, dropout=0.2
        # )
        # self.deconv2 = nn.ConvTranspose1d(
        #     32, 32, kernel_size=3, stride=2, padding=1
        # )
        # out_features = 64 + 32

        # self.rnn = nn.LSTM(
        #     80,
        #     32,
        #     num_layers=2,
        #     batch_first=True,
        #     bidirectional=True,
        # )

        # out_features = 256
        #################################################################

        #################################################################

    def forward(self, images, history_yaws = [], history_positions = [], speeds = []):
        # print("########x########", x.shape)
        x = self.backbone.conv1(images)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # print("########layer########")
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # print("########avgpool########")
        # print("#############x shape#############", x.shape)
        x = self.backbone.avgpool(x)
        # print("#############x shape#############", x.shape)
        # x = self.pool3d(x)
        x = torch.flatten(x, 1)
        # print("########head########", x.shape)
        # print("########head########", self.head)

        if len(history_yaws) != 0:
            speeds = torch.flatten(speeds, 1)
            history_yaws = torch.flatten(history_yaws, 1)
            history_positions = torch.flatten(history_positions, 1)
            

            
            # print("#############history_set shape#############", history_set.shape)

            

            # outs = []
            # features = self.resnet1(x)
            # out = self.deconv1(features)
            # # outs.append(out)
            # features = self.resnet2(out)
            # out = self.deconv2(features)
            # outs.append(out)
            # features = self.resnet3(out)
            # out = self.deconv3(features)
            # # outs.append(out)
            # features = self.resnet4(out)
            # out = self.deconv4(features)
            # outs.append(out)

            # out = out.permute(0, 2, 1)
            # features, _ = self.rnn(out)
            # for feature in aaaa:
            # print("#############features shape#############", features.shape)
            
            # features = torch.cat(outs, dim=1).permute(0, 2, 1)
            # features = torch.flatten(out, 1)
            # print("#############aaaa shape#############", features.shape)

            # features = self.history_set_head(history_set)
            # outs = []
            # feature, _ = self.yaws_rnn1(history_set)
            # out = self.yaws_deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            # outs.append(out)
            # feature, _ = self.yaws_rnn2(feature)
            # out = self.yaws_deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
            # outs.append(out)
            # feature = torch.cat(outs, dim=-1)
            # feature = nn.functional.normalize(feature, p=2, dim=2)




            # print("#############speeds shape#############", speeds)
            features = torch.cat((x, speeds, history_yaws, history_positions), dim=1)
            # x = x.permute(0, 2, 1)

            # outs = []
            # feature, _ = self.rnn1(x)
            # out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            # outs.append(out)
            # feature, _ = self.rnn2(feature)
            # out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
            # outs.append(out)
            # features, _ = self.rnn(x)
            # features = self.batchNorm(features)
            # features = self.dropOut(features)
            # print("#############out info#############", features.shape)

            # feature, _ = self.yaws_rnn1(history_yaws)
            # out = self.deconv1(feature.permute(0, 2, 1)).permute(0, 2, 1)
            # outs.append(out)
            # feature, _ = self.rnn2(feature)
            # out = self.deconv2(feature.permute(0, 2, 1)).permute(0, 2, 1)
            # outs.append(out)
            
            # for out in outs:
            #     print("#############out info#############", out.shape)
            # features = torch.cat(outs, dim=-1)

            # outs = torch.flatten(feature, 1)
            # outs = self.feture_linear(outs)

            # print("#############outs shape#############", outs.shape)
            # print("#############feature info#############", x.shape)

        
            # history_yaws_features = self.yaw_head(history_yaws)
            # history_position_features = self.position_head(history_position)
            # x = torch.cat((x, history_yaws_features, history_position_features), 1)

            
            # x = torch.cat((x, features), 1)

            # features = self.pool2d(features)
            # features = torch.flatten(features, 1)
            x = self.parend_head(features)
            # print("#############feature info#############", x.shape)
            x = x.view(x.shape[0], 50, -1)
            # print("#############feature info#############", x.shape)
            outs = []
            features, _ = self.rnn1(x)
            out = self.deconv1(features.permute(0,2,1)).permute(0,2,1)
            outs.append(out)
            features, _ = self.rnn2(features)
            out = self.deconv2(features.permute(0,2,1)).permute(0,2,1)
            outs.append(out)
            features = torch.cat(outs, dim=-1)
            # print("#############feature info#############", features.shape)
            features = self.batchNorm(features)
            # x = self.dropOut(x)
            x = torch.flatten(features, 1)
            # print("#############feature info#############", x.shape)

        else:
          x = self.head(x)
        # x = self.logit(x)
        
        preds = self.x_preds(x)
        modes = self.x_modes(x)
        return preds, modes
    

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        target_availabilities = batch["target_availabilities"]
        targets = batch["target_positions"]
        history_yaws = batch["history_yaws"]
        history_positions = batch["history_positions"]
        position_x = history_positions[:, :, 0]
        position_y = history_positions[:, :, 1]
        # print("#############tensor#############", batch["history_positions"].shape)
        # print("#############tensor#############", position_x[:, :-1].shape)
        # print("#############tensor#############", torch.unsqueeze(self.previous_postiion_x, 0).shape)
        speeds = torch.zeros(position_x.shape[0], position_x.shape[1], dtype=torch.float32, device=self.my_device)
        for i in range(position_x.shape[0]):
            sub_position_x = torch.cat((torch.unsqueeze(self.previous_postiion_x, 0),position_x[i, 1:]))
            sub_position_y = torch.cat((torch.unsqueeze(self.previous_postiion_x, 0),position_y[i, 1:]))
            select_position_x = torch.cat((position_x[i, :1], position_x[i, :-1]))
            select_position_y = torch.cat((position_y[i, :1], position_y[i, :-1]))
            speeds[i] = torch.add(torch.abs(torch.sub(sub_position_x, select_position_x)), torch.abs(torch.sub(sub_position_y, select_position_y)))
        self.previous_postiion_x = position_x[-1,-1]
        self.previous_postiion_y = position_y[-1,-1]
        speeds = torch.unsqueeze(speeds, -1)
        # self.previous_postiion_x = position_x[-1, -1, -1]
        # self.previous_postiion_y = position_y[-1, -1, -1]
        # speeds = torch.add(torch.abs(torch.sub(sub_position_x, position_x)), torch.abs(torch.sub(sub_position_y, position_y)))
        pred, confidences = self(images, history_yaws, history_positions, speeds)
        bs, _ = pred.shape
        preds = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        images = batch["image"]
        target_availabilities = batch["target_availabilities"]
        targets = batch["target_positions"]
        history_yaws = batch["history_yaws"]
        history_positions = batch["history_positions"]
        position_x = history_positions[:, :, 0]
        position_y = history_positions[:, :, 1]
        # print("#############tensor#############", batch["history_positions"].shape)
        # print("#############tensor#############", position_x[:, :-1].shape)
        # print("#############tensor#############", torch.unsqueeze(self.previous_postiion_x, 0).shape)
        speeds = torch.zeros(position_x.shape[0], position_x.shape[1], dtype=torch.float32, device=self.my_device)
        for i in range(position_x.shape[0]):
            sub_position_x = torch.cat((torch.unsqueeze(self.previous_postiion_x, 0),position_x[i, 1:]))
            sub_position_y = torch.cat((torch.unsqueeze(self.previous_postiion_x, 0),position_y[i, 1:]))
            select_position_x = torch.cat((position_x[i, :1], position_x[i, :-1]))
            select_position_y = torch.cat((position_y[i, :1], position_y[i, :-1]))
            speeds[i] = torch.add(torch.abs(torch.sub(sub_position_x, select_position_x)), torch.abs(torch.sub(sub_position_y, select_position_y)))
        self.previous_postiion_x = position_x[-1,-1]
        self.previous_postiion_y = position_y[-1,-1]
        speeds = torch.unsqueeze(speeds, -1)
        # self.previous_postiion_x = position_x[-1, -1, -1]
        # self.previous_postiion_y = position_y[-1, -1, -1]
        # speeds = torch.add(torch.abs(torch.sub(sub_position_x, position_x)), torch.abs(torch.sub(sub_position_y, position_y)))
        pred, confidences = self(images, history_yaws, history_positions, speeds)
        bs, _ = pred.shape
        preds = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)

        val_logs = {
            'val_loss': loss,
        }
        return val_logs
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_val_loss = torch.mean(torch.tensor([x['val_loss'] for x in validation_step_outputs])).item()
        # avg_loss = self.avg_loss

        # print(f'val_loss: {avg_val_loss}, loss: {avg_loss}')
        print(f'val_loss: {avg_val_loss}')

        # tensorboard_logs = {'val_loss': avg_val_loss, "loss": avg_loss}
        tensorboard_logs = {'val_loss': avg_val_loss}

        self.iterations.append(self.cur_epoch)
        # self.avg_losses.append(avg_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.times.append((time.time()-self.start)/60)
        self.start = time.time()
        
        results = pd.DataFrame({'iterations': self.iterations, 'avg_val_losses': self.avg_val_losses, 'elapsed_time (mins)': self.times})
        results.to_csv(f"{os.getcwd()}/result/test/{model_name}/results/train_metric{self.cur_epoch}.csv", index = False)

        self.cur_epoch+=1

        return {
            'val_loss': avg_val_loss,
            'log': tensorboard_logs,
        }

    

    def configure_optimizers(self):
        optimizer = AdamP(model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=1e-2)
       
        return optimizer
    
    def train_dataloader(self):
        return train_dataloader

    def val_dataloader(self):
        return val_dataloader




device_version = 1
model = LyftMultiModel(device_version)
model.to("cuda")

model_name = cfg["model_params"]["model_name"]
weight_path = cfg["model_params"]["weight_path"]
if weight_path:
    model.load_state_dict(torch.load(weight_path)['state_dict'])


trainer = Trainer(max_epochs=5, limit_val_batches=0.2, gpus=[device_version])

trainer.fit(model)




















