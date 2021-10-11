import numpy as np
import torch
from torch import nn
import SimpleModel.py

def load_model(path):
  model = SimpleModel().cuda()
  model.load_state_dict(torch.load(path))
  model.eval()
