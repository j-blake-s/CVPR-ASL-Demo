import torch
import numpy as np
import torch
from torch import nn

CLASSES = ["Tuesday","Bathroom","Name","Weight","Brown","Beer","Favorite","Colors","Hamburger","Marriage"]

class MaxPool(torch.nn.Module):
  def __init__(self):
    super().__init__()

    self.pool = torch.nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2))

  def forward(self, x):
    return self.pool(x)

def predict_sample(events):

  # Convert to Torch
  events = torch.from_numpy(events) # T C H W
  
  # Permute shape
  events = torch.permute(events, [1,2,3,0]) # C W H T
  
  # Max Pool
  events = MaxPool().forward(events) # C H/2 W/2 T/2
  events = torch.unsqueeze(events, dim=0).to(torch.float32)

  # Make Model
  model = torch.jit.load("./saved_models/model4.pt", map_location="cpu")
  model.eval()

  # Prediction
  pred = torch.argmax(model(events)[0])

  return CLASSES[pred]