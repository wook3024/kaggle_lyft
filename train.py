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
        'model_name': "train_resnest50+267+adamp+0.5+15+1e-4+threshold_0.5+history_yaws+history_positions",
        # 'model_name': "lr_finder_train",
        'lr': 1e-4,
        'weight_path': "./result/train/train_resnest50+267+adamp+0.5+15+1e-4+threshold_0.5/pytorch_lightning-models-v51.ckpt",
        # 'weight_path': "./model_resnet34_output_0.pth",
        # 'weight_path': None,
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
        'key': 'scenes/train.zarr',
        'batch_size': 48,
        'shuffle': True,
        'num_workers': 8
    },
    
    'test_data_loader': {
        'key': 'scenes/test.zarr',
        'batch_size': 48,
        'shuffle': False,
        'num_workers': 8
    },
    'val_data_loader': {
        'key': 'scenes/validate.zarr',
        'batch_size': 48,
        'shuffle': False,
        'num_workers': 8
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

# cur_step = 50000
# train_dataset = train_dataset[train_cfg["batch_size"] * 50000:]
train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                             num_workers=train_cfg["num_workers"], pin_memory=True)
val_dataloader = DataLoader(val_dataset, shuffle=val_cfg["shuffle"], batch_size=val_cfg["batch_size"], 
                             num_workers=val_cfg["num_workers"], pin_memory=True)                             
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
        self.step = 0
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
            os.mkdir(f'./result/train')
        except:
            pass
        try:
            os.mkdir(f'./result/train/{model_name}')
        except:
            pass
        try:
            os.mkdir(f'./result/train/{model_name}/results')
        except:
            pass
        try:
            os.mkdir(f'./result/train/{model_name}/models')
        except:
            pass
        try:
            os.mkdir(f'./result/train/{model_name}/logs')
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
        self.add_features_head = nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(in_features=backbone_out_features+33, out_features=4096),
        )
        # self.head = nn.Sequential(
        #     nn.Linear(in_features=backbone_out_features, out_features=4096),
        # )
        
        self.num_preds = num_targets * self.num_modes

        self.x_preds = nn.Linear(4096, out_features=self.num_preds)
        self.x_modes = nn.Linear(4096, out_features=self.num_modes)

        # self.logit = nn.Linear(4096, out_features=self.num_preds + self.num_modes)

    def forward(self, images, history_yaws, history_positions):
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
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        # print("########head########", x.shape)
        # print("########head########", self.head)
        history_yaws = torch.flatten(history_yaws, 1)
        history_positions = torch.flatten(history_positions, 1)
        features = torch.cat((x, history_yaws, history_positions), dim=1)

        
        features = self.add_features_head(features)
        # x = self.logit(x)
        
        preds = self.x_preds(features)
        modes = self.x_modes(features)
        # print("########pred, modes########", pred.shape, modes.shape)
        # pred (batch_size)x(modes)x(time)x(2D coords)
        # confidences (batch_size)x(modes)
        
        # bs, _ = x.shape
        # pred, confidences = torch.split(x, self.num_preds, dim=1)
        # pred = pred.view(bs, self.num_modes, self.future_len, 2)
        # assert confidences.shape == (bs, self.num_modes)
        # confidences = torch.softmax(confidences, dim=1)
        # return pred, confidences
        return preds, modes
    

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        target_availabilities = batch["target_availabilities"]
        targets = batch["target_positions"]
        history_yaws = torch.mul(batch["history_yaws"], 1.6)
        history_positions = batch["history_positions"]
        history_positions[:,:,0] = torch.div(history_positions[:,:,0],6.0)
        history_positions[:,:,1] = torch.div(history_positions[:,:,1],1.9)
        
        pred, confidences = self(images, history_yaws, history_positions)
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
        history_yaws = torch.mul(batch["history_yaws"], 1.6)
        history_positions = batch["history_positions"]
        history_positions[:,:,0] = torch.div(history_positions[:,:,0],6.0)
        history_positions[:,:,1] = torch.div(history_positions[:,:,1],1.9)
        
        pred, confidences = self(images, history_yaws, history_positions)
        bs, _ = pred.shape
        preds = pred.view(bs, self.num_modes, self.future_len, 2)
        assert confidences.shape == (bs, self.num_modes)
        confidences = torch.softmax(confidences, dim=1)
        loss = pytorch_neg_multi_log_likelihood_batch(targets, preds, confidences, target_availabilities)

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

        # torch.save(model.state_dict(), f'{os.getcwd()}/result/train/{model_name}/models/save_model_{self.cur_epoch}.pth')
        self.iterations.append(self.cur_epoch)
        # self.avg_losses.append(avg_loss)
        self.avg_val_losses.append(avg_val_loss)
        self.times.append((time.time()-self.start)/60)
        self.start = time.time()
        
        results = pd.DataFrame({'iterations': self.iterations, 'avg_val_losses': self.avg_val_losses, 'elapsed_time (mins)': self.times})
        results.to_csv(f"{os.getcwd()}/result/train/{model_name}/results/train_metric{self.cur_epoch}.csv", index = False)

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
    filepath=Path(f'{os.getcwd()}/result/train/{model_name}/models'),
    save_top_k=1000000,
    monitor='val_loss',
    verbose=True,
    prefix='pytorch_lightning'
)

logger = TensorBoardLogger(
    save_dir=Path(f'{os.getcwd()}/result/train/{model_name}/logs'),
    version=1,
    name='lightning_logs'
)

# model = LyftMultiModel()
weight_path = cfg["model_params"]["weight_path"]
model = LyftMultiModel.load_from_checkpoint(weight_path, strict=False)
model.to("cuda")

# if weight_path:
#     model.load_state_dict(torch.load(weight_path)['state_dict'])

trainer = Trainer(logger=logger, 
                  checkpoint_callback=checkpoint_callback, 
                  val_check_interval=10000,
                  limit_val_batches=1000, 
                  gpus=2, 
                  accelerator='ddp')

trainer.fit(model)

















