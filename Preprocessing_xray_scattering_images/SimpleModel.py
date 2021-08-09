import torch.nn as nn


class SimpleModel(nn.Module):

    def __init__(self, p1=0.1, p2=0.0, hid_dim1=512, hid_dim2=128, in_dim: int = bin_size, out_dim: int = 2):
        super().__init__()
        self.leaky_relu = nn.LeakyReLU(0.02, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.model = nn.Sequential(
            nn.Linear(in_dim, hid_dim1),
            nn.Dropout(p1),
            self.leaky_relu,
            nn.Linear(hid_dim1, hid_dim2),
            nn.Dropout(p2),
            self.leaky_relu,
            nn.Linear(hid_dim2, out_dim))

    def forward(self, x):
        return self.model(x)

    def cpu_state_dict(self):
        return OrderedDict([(k, v.to('cpu')) for k, v in self.state_dict().items()])
