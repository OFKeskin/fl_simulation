from torch import nn
import torch.nn.functional as F 


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(MLP, self).__init__()
        self.layer = nn.Linear(dim_in, dim_out)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return self.softmax(x)