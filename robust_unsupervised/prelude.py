from typing import *

import copy
import os

import functools
import sys
import torch.optim as optim
import tqdm
import dataclasses
import numpy as np
import PIL.Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import dnnlib as dnnlib
import dnnlib.legacy as legacy
# import legacy
import shutil
from functools import partial
import itertools
import warnings
from warnings import warn
import datetime
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
import training.networks as networks

from abc import ABC, abstractmethod, abstractstaticmethod, abstractclassmethod
from dataclasses import dataclass, field

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

warnings.filterwarnings("ignore", r"Named tensors and all their associated APIs.*")
warnings.filterwarnings("ignore", r"Arguments other than a weight enum.*")
warnings.filterwarnings("ignore", r"The parameter 'pretrained' is deprecated.*")



