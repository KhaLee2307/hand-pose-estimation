r"""
Network utilities
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def BHWC_to_BCHW(x):
    """
    :param x: torch tensor, B x H x W x C
    :return:  torch tensor, B x C x H x W
    """
    return x.unsqueeze(1).transpose(1, -1).squeeze(-1)

def normalize_image(im):
    """
    byte -> float, / pixel_max, - 0.5
    :param im: torch byte tensor, B x C x H x W, 0 ~ 255
    :return:   torch float tensor, B x C x H x W, -0.5 ~ 0.5
    """
    return ((im.float() / 255.0) - 0.5)