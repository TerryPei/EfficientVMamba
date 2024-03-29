import torch

import os
from functools import partial
from typing import Callable

import torch
from torch import nn
from torch.utils import checkpoint

from mmengine.model import BaseModule
from mmdet.registry import MODELS as MODELS_MMDET
from mmseg.registry import MODELS as MODELS_MMSEG

def import_abspy(name="models", path="classification/"):
    import sys
    import importlib
    path = os.path.abspath(path)
    assert os.path.isdir(path)
    sys.path.insert(0, path)
    module = importlib.import_module(name)
    sys.path.pop(0)
    return module

build = import_abspy(
    "models", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)
Backbone_VSSM: nn.Module = build.vmamba.Backbone_VSSM
Backbone_EfficientVSSM: nn.Module = build.vmamba_efficient.Backbone_EfficientVSSM

ckpt = 'ckpts/mobilebase/efficient_vmamba_tiny.ckpt'
checkpoint = torch.load(ckpt, map_location='cpu')
print(checkpoint.keys())

efficient_vssm_tiny = Backbone_EfficientVSSM(pretrained="ckpts/mobilebase/efficient_vmamba_tiny.ckpt")