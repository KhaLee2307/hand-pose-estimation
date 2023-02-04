#Network utilities
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path as osp
from collections import OrderedDict

import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, numIn, numOut):
        super(Residual, self).__init__()
        self.numIn = numIn
        self.numOut = numOut
        self.bn = nn.BatchNorm2d(self.numIn)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(self.numIn, self.numOut // 2, bias=True, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(self.numOut // 2)
        self.conv2 = nn.Conv2d(self.numOut // 2, self.numOut // 2, bias=True, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(self.numOut // 2)
        self.conv3 = nn.Conv2d(self.numOut // 2, self.numOut, bias=True, kernel_size=1)

        if self.numIn != self.numOut:
            self.conv4 = nn.Conv2d(self.numIn, self.numOut, bias=True, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.numIn != self.numOut:
            residual = self.conv4(x)

        return out + residual

def load_net_model(model_path, net):
    assert (osp.isfile(model_path)), ('The model does not exist! Error path:\n%s' % model_path)

    model_dict = torch.load(model_path, map_location='cpu')
    module_prefix = 'module.'
    module_prefix_len = len(module_prefix)

    for k in model_dict.keys():
        if k[:module_prefix_len] != module_prefix:
            net.load_state_dict(model_dict)
            return 0

    del_keys = filter(lambda x: 'num_batches_tracked' in x, model_dict.keys())
    for k in del_keys:
        del model_dict[k]

    model_dict = OrderedDict([(k[module_prefix_len:], v) for k, v in model_dict.items()])
    net.load_state_dict(model_dict)
    return 0