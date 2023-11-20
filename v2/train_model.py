import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import argparse


class Net(nn.Module):
    def __init__(self, n_concat, n_freq, n_out):
        super(Net, self).__init__()
        n_in = n_concat * n_freq
        n_hid = 500

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, n_hid)
        self.fc4 = nn.Linear(n_hid, n_out)

    def forward(self, x):
        drop_p = 0.2
        x1 = x.view(len(x), -1)
        x2 = F.dropout(F.relu(self.fc1(x1)), p=drop_p, training=self.training)
        x3 = F.dropout(F.relu(self.fc2(x2)), p=drop_p, training=self.training)
        x4 = F.dropout(F.relu(self.fc3(x3)), p=drop_p, training=self.training)
        x5 = F.sigmoid(self.fc4(x4))
        return x5
    
def train(args):
    cuda = args.use_cuda and torch.cuda.is_available()
    print("cuda:", cuda)
    

