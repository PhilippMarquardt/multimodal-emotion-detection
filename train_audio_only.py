#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import numpy as np
from collections import Counter
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset

# __all__ = ['ResNeXt', 'resnet50', 'resnet101']


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1, downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNeXt(nn.Module):
    def __init__(
        self,
        block,
        layers,
        sample_size,
        sample_duration,
        shortcut_type="B",
        cardinality=32,
        num_classes=400,
        last_fc=True,
    ):
        self.last_fc = last_fc

        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 128, layers[0], shortcut_type, cardinality
        )
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2
        )
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2
        )
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2
        )
        last_duration = math.ceil(sample_duration / 16)
        last_size = math.ceil(sample_size / 32)
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(cardinality * 32 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, cardinality, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.last_fc:
            x = self.fc(x)

        return x


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append("layer{}".format(ft_begin_index))
    ft_module_names.append("fc")

    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({"params": v})
                break
        else:
            parameters.append({"params": v, "lr": 0.0})

    return parameters


def resnet50(**kwargs):
    """Constructs a ResNet-50 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 6, 3], **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model."""
    model = ResNeXt(ResNeXtBottleneck, [3, 8, 36, 3], **kwargs)
    return model


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

    def forward(self, x):

        net = x

        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.conv(net)

        return net


class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding

    params:
        kernel_size: kernel size
        stride: the stride of the window. Default value is kernel_size

    input: (n_sample, n_channel, n_length)
    """

    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):

        net = x

        # compute pad shape
        p = max(0, self.kernel_size - 1)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)

        net = self.max_pool(net)

        return net


class Swish(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


class BasicBlock(nn.Module):
    """
    Basic Block:
        conv1 -> convk -> conv1

    params:
        in_channels: number of input channels
        out_channels: number of output channels
        ratio: ratio of channels to out_channels
        kernel_size: kernel window length
        stride: kernel step size
        groups: number of groups in convk
        downsample: whether downsample length
        use_bn: whether use batch_norm
        use_do: whether use dropout

    input: (n_sample, in_channels, n_length)
    output: (n_sample, out_channels, (n_length+stride-1)//stride)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ratio,
        kernel_size,
        stride,
        groups,
        downsample,
        is_first_block=False,
        use_bn=True,
        use_do=True,
    ):
        super(BasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.downsample = downsample
        self.stride = stride if self.downsample else 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.middle_channels = int(self.out_channels * self.ratio)

        # the first conv, conv1
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.activation1 = Swish()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=self.in_channels,
            out_channels=self.middle_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

        # the second conv, convk
        self.bn2 = nn.BatchNorm1d(self.middle_channels)
        self.activation2 = Swish()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=self.middle_channels,
            out_channels=self.middle_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            groups=self.groups,
        )

        # the third conv, conv1
        self.bn3 = nn.BatchNorm1d(self.middle_channels)
        self.activation3 = Swish()
        self.do3 = nn.Dropout(p=0.5)
        self.conv3 = MyConv1dPadSame(
            in_channels=self.middle_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            groups=1,
        )

        # Squeeze-and-Excitation
        r = 2
        self.se_fc1 = nn.Linear(self.out_channels, self.out_channels // r)
        self.se_fc2 = nn.Linear(self.out_channels // r, self.out_channels)
        self.se_activation = Swish()

        if self.downsample:
            self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):

        identity = x

        out = x
        # the first conv, conv1
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.activation1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)

        # the second conv, convk
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)

        # the third conv, conv1
        if self.use_bn:
            out = self.bn3(out)
        out = self.activation3(out)
        if self.use_do:
            out = self.do3(out)
        out = self.conv3(out)  # (n_sample, n_channel, n_length)

        # Squeeze-and-Excitation
        se = out.mean(-1)  # (n_sample, n_channel)
        se = self.se_fc1(se)
        se = self.se_activation(se)
        se = self.se_fc2(se)
        se = F.sigmoid(se)  # (n_sample, n_channel)
        out = torch.einsum("abc,ab->abc", out, se)

        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)

        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1, -2)
            ch1 = (self.out_channels - self.in_channels) // 2
            ch2 = self.out_channels - self.in_channels - ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1, -2)

        # shortcut
        out += identity

        return out


class BasicStage(nn.Module):
    """
    Basic Stage:
        block_1 -> block_2 -> ... -> block_M
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        ratio,
        kernel_size,
        stride,
        groups,
        i_stage,
        m_blocks,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super(BasicStage, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio
        self.kernel_size = kernel_size
        self.groups = groups
        self.i_stage = i_stage
        self.m_blocks = m_blocks
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        self.block_list = nn.ModuleList()
        for i_block in range(self.m_blocks):

            # first block
            if self.i_stage == 0 and i_block == 0:
                self.is_first_block = True
            else:
                self.is_first_block = False
            # downsample, stride, input
            if i_block == 0:
                self.downsample = True
                self.stride = stride
                self.tmp_in_channels = self.in_channels
            else:
                self.downsample = False
                self.stride = 1
                self.tmp_in_channels = self.out_channels

            # build block
            tmp_block = BasicBlock(
                in_channels=self.tmp_in_channels,
                out_channels=self.out_channels,
                ratio=self.ratio,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=self.groups,
                downsample=self.downsample,
                is_first_block=self.is_first_block,
                use_bn=self.use_bn,
                use_do=self.use_do,
            )
            self.block_list.append(tmp_block)

    def forward(self, x):

        out = x

        for i_block in range(self.m_blocks):
            net = self.block_list[i_block]
            out = net(out)
            if self.verbose:
                print(
                    "stage: {}, block: {}, in_channels: {}, out_channels: {}, outshape: {}".format(
                        self.i_stage,
                        i_block,
                        net.in_channels,
                        net.out_channels,
                        list(out.shape),
                    )
                )
                print(
                    "stage: {}, block: {}, conv1: {}->{} k={} s={} C={}".format(
                        self.i_stage,
                        i_block,
                        net.conv1.in_channels,
                        net.conv1.out_channels,
                        net.conv1.kernel_size,
                        net.conv1.stride,
                        net.conv1.groups,
                    )
                )
                print(
                    "stage: {}, block: {}, convk: {}->{} k={} s={} C={}".format(
                        self.i_stage,
                        i_block,
                        net.conv2.in_channels,
                        net.conv2.out_channels,
                        net.conv2.kernel_size,
                        net.conv2.stride,
                        net.conv2.groups,
                    )
                )
                print(
                    "stage: {}, block: {}, conv1: {}->{} k={} s={} C={}".format(
                        self.i_stage,
                        i_block,
                        net.conv3.in_channels,
                        net.conv3.out_channels,
                        net.conv3.kernel_size,
                        net.conv3.stride,
                        net.conv3.groups,
                    )
                )

        return out


class Net1D(nn.Module):
    """

    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)

    Output:
        out: (n_samples)

    params:
        in_channels
        base_filters
        filter_list: list, filters for each stage
        m_blocks_list: list, number of blocks of each stage
        kernel_size
        stride
        groups_width
        n_stages
        n_classes
        use_bn
        use_do

    """

    def __init__(
        self,
        in_channels,
        base_filters,
        ratio,
        filter_list,
        m_blocks_list,
        kernel_size,
        stride,
        groups_width,
        n_classes,
        use_bn=True,
        use_do=True,
        verbose=False,
    ):
        super(Net1D, self).__init__()

        self.in_channels = in_channels
        self.base_filters = base_filters
        self.ratio = ratio
        self.filter_list = filter_list
        self.m_blocks_list = m_blocks_list
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups_width = groups_width
        self.n_stages = len(filter_list)
        self.n_classes = n_classes
        self.use_bn = use_bn
        self.use_do = use_do
        self.verbose = verbose

        # first conv
        self.first_conv = MyConv1dPadSame(
            in_channels=in_channels,
            out_channels=self.base_filters,
            kernel_size=self.kernel_size,
            stride=2,
        )
        self.first_bn = nn.BatchNorm1d(base_filters)
        self.first_activation = Swish()

        # stages
        self.stage_list = nn.ModuleList()
        in_channels = self.base_filters
        for i_stage in range(self.n_stages):

            out_channels = self.filter_list[i_stage]
            m_blocks = self.m_blocks_list[i_stage]
            tmp_stage = BasicStage(
                in_channels=in_channels,
                out_channels=out_channels,
                ratio=self.ratio,
                kernel_size=self.kernel_size,
                stride=self.stride,
                groups=out_channels // self.groups_width,
                i_stage=i_stage,
                m_blocks=m_blocks,
                use_bn=self.use_bn,
                use_do=self.use_do,
                verbose=self.verbose,
            )
            self.stage_list.append(tmp_stage)
            in_channels = out_channels

        # final prediction
        self.dense = nn.Linear(in_channels, n_classes)

    def forward(self, x):

        out = x

        # first conv
        out = self.first_conv(out)
        if self.use_bn:
            out = self.first_bn(out)
        out = self.first_activation(out)

        # stages
        for i_stage in range(self.n_stages):
            net = self.stage_list[i_stage]
            out = net(out)

        # final prediction
        out = out.mean(-1)
        out = self.dense(out)

        return out


import torch
from torch import nn

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

# helpers


def exists(val):
    return val is not None


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        image_patch_size,
        frames,
        frame_patch_size,
        num_classes,
        dim,
        spatial_depth,
        temporal_depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        assert (
            frames % frame_patch_size == 0
        ), "Frames must be divisible by frame patch size"

        num_image_patches = (image_height // patch_height) * (
            image_width // patch_width
        )
        num_frame_patches = frames // frame_patch_size

        patch_dim = channels * patch_height * patch_width * frame_patch_size

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.global_average_pool = pool == "mean"

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (f pf) (h p1) (w p2) -> b f (h w) (p1 p2 pf c)",
                p1=patch_height,
                p2=patch_width,
                pf=frame_patch_size,
            ),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frame_patches, num_image_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)

        self.spatial_cls_token = (
            nn.Parameter(torch.randn(1, 1, dim))
            if not self.global_average_pool
            else None
        )
        self.temporal_cls_token = (
            nn.Parameter(torch.randn(1, 1, dim))
            if not self.global_average_pool
            else None
        )

        self.spatial_transformer = Transformer(
            dim, spatial_depth, heads, dim_head, mlp_dim, dropout
        )
        self.temporal_transformer = Transformer(
            dim, temporal_depth, heads, dim_head, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, video):
        x = self.to_patch_embedding(video)
        b, f, n, _ = x.shape

        x = x + self.pos_embedding[:, :f, :n]

        if exists(self.spatial_cls_token):
            spatial_cls_tokens = repeat(
                self.spatial_cls_token, "1 1 d -> b f 1 d", b=b, f=f
            )
            x = torch.cat((spatial_cls_tokens, x), dim=2)

        x = self.dropout(x)

        x = rearrange(x, "b f n d -> (b f) n d")

        # attend across space

        x = self.spatial_transformer(x)

        x = rearrange(x, "(b f) n d -> b f n d", b=b)

        # excise out the spatial cls tokens or average pool for temporal attention

        x = (
            x[:, :, 0]
            if not self.global_average_pool
            else reduce(x, "b f n d -> b f d", "mean")
        )

        # append temporal CLS tokens

        if exists(self.temporal_cls_token):
            temporal_cls_tokens = repeat(self.temporal_cls_token, "1 1 d-> b 1 d", b=b)

            x = torch.cat((temporal_cls_tokens, x), dim=1)

        # attend across time

        x = self.temporal_transformer(x)

        # excise out temporal cls token or average pool

        x = (
            x[:, 0]
            if not self.global_average_pool
            else reduce(x, "b f d -> b d", "mean")
        )

        x = self.to_latent(x)
        return self.mlp_head(x)


# In[2]:


import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchvision.transforms import Lambda
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.io import write_video
from torchvision.transforms import Resize
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

# from vit_pytorch.vivit import ViT
import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchaudio.transforms import MelSpectrogram
from moviepy.editor import VideoFileClip, AudioFileClip
import torchaudio
import subprocess
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from torch.utils.data.dataset import random_split

torch.manual_seed(42)


class CustomDataset(Dataset):
    def __init__(self, root_dir, num_frames=64, skip_frames=1):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.class_names = sorted(os.listdir(root_dir))
        self.file_paths = []
        self.labels = []
        self.skip_frames = skip_frames
        for label, class_name in enumerate(self.class_names):
            class_folder = os.path.join(root_dir, class_name)
            file_names = os.listdir(class_folder)

            for file_name in file_names:
                file_path = os.path.join(class_folder, file_name)
                self.file_paths.append(file_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        video_path = self.file_paths[index]
        # print(video_path)
        label = self.labels[index]
        video, audio, info = read_video(video_path, pts_unit="sec")
        sample_rate = info.get("audio_fps")
        spectrogram_transform = MelSpectrogram(sample_rate=sample_rate)
        audio = audio.mean(dim=0)  # Convert stereo to mono if necessary
        audio_raw = audio.clone()  # Save a copy of raw audio

        # Sample frames
        total_frames = video.shape[0]
        selected_frames = []
        start_frame = 0
        take_middle = False
        if take_middle:
            # Always sample from the middle if `take_middle` is True
            middle_frame = total_frames // 2
            start_frame = middle_frame - (self.num_frames * self.skip_frames // 2)
            # Handle case when start frame is negative (i.e., not enough frames to satisfy the requirement)
            if start_frame < 0:
                padding = abs(start_frame)
                start_frame = 0
                selected_frames = torch.cat(
                    [
                        torch.zeros(padding, *video.shape[1:]),
                        video[: self.num_frames * self.skip_frames + padding],
                    ]
                )
            else:
                frame_indices = torch.arange(
                    start_frame,
                    start_frame + self.num_frames * self.skip_frames,
                    self.skip_frames,
                )
            selected_frames = video[frame_indices]
        else:
            # print(total_frames)
            if total_frames < self.num_frames * self.skip_frames:
                # Pad with zeros if the video has fewer frames than required
                selected_frames = video
                padding = self.num_frames * self.skip_frames - total_frames
                selected_frames = torch.cat(
                    [selected_frames, torch.zeros(padding, *video.shape[1:])]
                )
            else:
                # print("asd")
                # Randomly sample `num_frames` frames from the video
                start_frame = torch.randint(
                    0, total_frames - self.num_frames * self.skip_frames + 1, ()
                )
                frame_indices = torch.arange(
                    start_frame,
                    start_frame + self.num_frames * self.skip_frames,
                    self.skip_frames,
                )
                selected_frames = video[frame_indices]
        selected_frames = selected_frames.float() / 255.0
        if audio_raw.dim() == 1:
            audio_raw = audio_raw.unsqueeze(0)
        audio_samples_per_frame = sample_rate / info["video_fps"]

        # Calculate the indices of the audio samples corresponding to the selected video frames
        audio_start_sample = int(audio_samples_per_frame * start_frame)
        audio_end_sample = int(
            audio_samples_per_frame * (start_frame + self.num_frames * self.skip_frames)
        )

        # Select the corresponding audio samples
        audio_raw = audio_raw[:, audio_start_sample:audio_end_sample]

        # Generate the MelSpectrogram from the selected raw audio
        spectrogram = spectrogram_transform(audio_raw)
        # resized_spectrogram = F.interpolate(spectrogram.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
        # spectrogram = spectrogram.unsqueeze(0)  # Add a channel dimension
        spectrogram = (spectrogram - spectrogram.min()) / (
            spectrogram.max() - spectrogram.min()
        )  # Normalize to [0, 1]
        # print("Spectogramshaoe", spectrogram.shape)
        #         spectrogram_resize_transform = Resize(size=(self.resize_longest, self.resize_longest))
        #         spectrogram = spectrogram_resize_transform(spectrogram)

        # resize the spectrogram to 224x224
        spectrogram = F.interpolate(
            spectrogram.unsqueeze(0),
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        )
        # print(audio_raw.shape)
        # print(spectrogram.shape)
        # padding_multiple = 2
        # num_mels, time_frames = spectrogram.shape[-2], spectrogram.shape[-1]

        # # Compute padding sizes
        # pad_mels = (padding_multiple - (num_mels % padding_multiple)) % padding_multiple
        # pad_time_frames = (
        #     padding_multiple - (time_frames % padding_multiple)
        # ) % padding_multiple

        # # Pad the spectrogram
        # spectrogram = torch.nn.functional.pad(
        #     spectrogram, (0, pad_time_frames, 0, pad_mels)
        # )
        # return (
        #     selected_frames[0 : self.num_frames, :, :].permute(3, 0, 1, 2).float(),
        #     spectrogram.repeat(3, 1, 1),
        #     audio_raw,
        #     sample_rate,
        #     label,
        # )
        # print(audio_raw.max())
        # print(audio_raw.min())
        return (
            selected_frames.permute(3, 0, 1, 2).float(),
            audio_raw,
            label,
        )


if __name__ == "__main__":
    dataset = CustomDataset(root_dir=r"dataset/")
    # dataloader = torch.utils.data.DataLoader(
    #     dataset, batch_size=8, shuffle=True, num_workers=4
    # )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create the dataloaders
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=6
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )
    # print(len(dataset))
    # for x, spectogram, audio_raw, sample_rate, y in dataloader:
    #     print(x.shape)
    #     print(spectogram.shape)
    #     video_path = "output_video.mp4"
    #     audio_path = "output_audio.wav"
    #     final_output = "output.mp4"

    #     # Save video
    #     write_video(video_path, x[0].permute(1, 2, 3, 0), fps=30)
    #     break
    # print(spectogram.shape)
    # cv2.imwrite("xd.png", spectogram[0].permute(1,2,0).numpy() * 255.)
    # plt.imsave('xdddsad.png', spectogram[0].permute(1,2,0).numpy().astype(float), cmap='viridis')
    # Save audio with the correct sample rate
    # torchaudio.save(audio_path, audio_raw[0], sample_rate=int(sample_rate))

    # Merge video and audio
    # command = ['ffmpeg', '-y', '-i', video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', final_output]
    # subprocess.run(command, check=True)

    # In[ ]:

    import torch
    import torchvision.models as models
    import torch.nn as nn
    from torch.nn import functional as F
    from tqdm import tqdm

    import torch
    import torchvision.models as models
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    from torch.utils.tensorboard import SummaryWriter

    class VideoNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Pretrained models for video and audio processing
            self.image_model = models.resnet50(pretrained=True)

            # Replace the final fully connected layer of the audio model
            num_ftrs_audio = self.image_model.fc.in_features
            self.image_model.fc = nn.Linear(num_ftrs_audio, 512)

            # MLP for combining features from the two streams
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 8)  # Assuming 8 classes
            encoder_layers = TransformerEncoderLayer(d_model=512, nhead=8)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=4)

        def forward(self, video, audio_spectrogram):
            batch_size, c, num_frames, h, w = video.shape

            # Empty tensor to store the embeddings for each frame
            frame_embeddings = torch.empty(
                (batch_size, num_frames, 512), device=video.device
            )

            for i in range(num_frames):
                frame = video[:, :, i, :, :]
                frame_embedding = self.image_model(frame)
                frame_embeddings[:, i, :] = frame_embedding

            # Transpose for transformer encoder: [num_frames, batch_size, 512]
            frame_embeddings = frame_embeddings.transpose(0, 1)

            output = self.transformer_encoder(
                frame_embeddings
            )  # [num_frames, batch_size, 512]
            output = output.transpose(0, 1)

            # Take the mean across frames
            output = output.mean(dim=1)  # [batch_size, 512]

            # Pass through the linear layer
            output = self.fc2(output)  # [batch_size, num_classes]
            return output

    class VideoNetLSTM(nn.Module):
        def __init__(self):
            super().__init__()

            self.image_model = models.resnet50(pretrained=True)
            num_ftrs_image = self.image_model.fc.in_features
            self.image_model.fc = nn.Linear(
                num_ftrs_image, 512
            )  # Output shape of each frame: [batch_size, 512]

            self.bilstm = nn.LSTM(
                512, 256, batch_first=True, bidirectional=True
            )  # Input size: 512, hidden size: 256

            # Linear layers
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 8)  # Assuming 8 classes

        def forward(self, video, audio):
            batch_size, c, num_frames, h, w = video.shape

            # Empty tensor to store the embeddings for each frame
            frame_embeddings = torch.empty(
                (batch_size, num_frames, 512), device=video.device
            )

            for i in range(num_frames):
                frame = video[:, :, i, :, :]
                frame_embedding = self.image_model(frame)
                frame_embeddings[:, i, :] = frame_embedding

            # Pass frame embeddings through BiLSTM
            output, _ = self.bilstm(frame_embeddings)

            # Take the mean across frames
            output = output.mean(dim=1)  # [batch_size, hidden_size * 2]

            # Pass through fc1 and fc2
            output = nn.functional.relu(self.fc1(output))
            output = self.fc2(output)

            return output

    class AVNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Pretrained models for video and audio processing
            self.video_model = models.video.mvit_v2_s(pretrained=True)
            # self.video_model = resnet50(
            #     last_fc=False, sample_duration=16, sample_size=224
            # )
            # self.video_model = ViT(
            #     image_size=224,  # image size
            #     frames=64,  # number of frames
            #     image_patch_size=16,  # image patch size
            #     frame_patch_size=8,  # frame patch size
            #     num_classes=512,  # number of classes
            #     dim=256,  # dimension of the transformer
            #     spatial_depth=2,  # depth of the spatial transformer
            #     temporal_depth=2,  # depth of the temporal transformer
            #     heads=1,  # number of attention heads
            #     mlp_dim=1024,  # dimension of MLP
            # )
            # self.audio_model = models.resnet50(pretrained=True)

            # Replace the final fully connected layer of the audio model
            # num_ftrs_audio = self.audio_model.fc.in_features
            # self.audio_model.fc = nn.Linear(num_ftrs_audio, 512)

            # Replace the final layer of the video model
            # num_ftrs_video = self.video_model.head[1].in_features
            # self.video_model.head[1] = nn.Linear(num_ftrs_video, 512)

            # MLP for combining features from the two streams
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 8)  # Assuming 8 classes

        def forward(self, video, audio_spectrogram):
            # Process video and audio
            # print(video.shape)
            video_outputs = self.video_model(video)
            # print(video_outputs.shape)
            # audio_outputs = self.audio_model(audio_spectrogram)

            # Concatenate features from the two streams
            # combined = torch.cat((video_outputs, audio_outputs), dim=1)

            # Pass the combined features through the MLP
            combined = F.relu(self.fc1(video_outputs))
            outputs = self.fc2(combined)

            return outputs

    class AVNetAudio(nn.Module):
        def __init__(self):
            super().__init__()
            # Pretrained models for video and audio processing
            self.video_model = models.video.mvit_v2_s(pretrained=True)
            # self.video_model = resnet50(
            #     last_fc=False, sample_duration=16, sample_size=224
            # )
            # self.video_model = ViT(
            #     image_size=224,  # image size
            #     frames=64,  # number of frames
            #     image_patch_size=16,  # image patch size
            #     frame_patch_size=8,  # frame patch size
            #     num_classes=512,  # number of classes
            #     dim=256,  # dimension of the transformer
            #     spatial_depth=2,  # depth of the spatial transformer
            #     temporal_depth=2,  # depth of the temporal transformer
            #     heads=1,  # number of attention heads
            #     mlp_dim=1024,  # dimension of MLP
            # )
            base_filters = 64
            filter_list = [64, 160, 160, 400, 400, 1024, 1024]
            m_blocks_list = [2, 2, 2, 3, 3, 4, 4]
            self.audio_model = model = Net1D(
                in_channels=1,
                base_filters=base_filters,
                ratio=1.0,
                filter_list=filter_list,
                m_blocks_list=m_blocks_list,
                kernel_size=16,
                stride=2,
                groups_width=16,
                verbose=False,
                n_classes=512,
            )

            # Replace the final fully connected layer of the audio model
            # num_ftrs_audio = self.audio_model.fc.in_features
            # self.audio_model.fc = nn.Linear(num_ftrs_audio, 512)

            # Replace the final layer of the video model
            # num_ftrs_video = self.video_model.head[1].in_features
            # self.video_model.head[1] = nn.Linear(num_ftrs_video, 512)

            # MLP for combining features from the two streams
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 8)  # Assuming 8 classes

        def forward(self, video, audio):
            # Process video and audio
            # print(video.shape)
            # video_outputs = self.video_model(video)
            # audio = audio.permute(0, 2, 1)
            # print(audio.shape)
            audio_outputs = self.audio_model(audio)
            # print(audio_outputs.shape)
            # print(video_outputs.shape)
            # audio_outputs = self.audio_model(audio_spectrogram)

            # Concatenate features from the two streams
            # combined = torch.cat((video_outputs, audio_outputs), dim=1)

            # Pass the combined features through the MLP
            combined = F.relu(self.fc1(audio_outputs))
            outputs = self.fc2(combined)

            return outputs

    model = AVNetAudio()
    model.cuda()
    work_dir = "onlyaudio64frames"
    os.mkdir(work_dir)

    def contains_invalid_values(tensor):
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    # model.load_state_dict(torch.load("model_epochss.pth"))
    num_epochs = 500  # Example number of epochs, adjust as needed
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=0.01)
    writer = SummaryWriter(os.path.join(work_dir, "runs/lstmmultilabel"))
    best_val_accuracy = 0.0  # Initialise with 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        pbar = tqdm(
            total=len(dataloader),
            desc=f"Epoch [{epoch+1}/{num_epochs}]",
            dynamic_ncols=True,
        )
        acc_batch_epoch = []
        loss_batch_epoch = []
        for i, (video, audio, labels) in enumerate(dataloader):
            video_path = f"output/{i}.mp4"
            # write_video(video_path, video[0].permute(1,2,3,0), fps=30)
            video = video.cuda()
            audio = audio.cuda()
            # if contains_invalid_values(audio_spectrogram):
            #    print("Broken")
            #    continue
            #         print(video.shape)
            #         print(contains_invalid_values(audio_spectrogram))
            #         print(torch.max(audio_spectrogram))
            #         print(torch.min(audio_spectrogram))
            # audio_spectrogram = audio_spectrogram.cuda()

            labels = labels.cuda()

            # Forward pass
            outputs = model(video, audio)

            loss = criterion(outputs, labels)

            # Backward pass and optimization

            loss.backward()
            optimizer.step()
            model.zero_grad()
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

            avg_acc = (correct_predictions / total_samples) * 100
            acc_batch_epoch.append(avg_acc)
            loss_batch_epoch.append(loss.item())

            # Update progress bar description
            pbar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(loss_batch_epoch):.4f}, Acc: {np.mean(acc_batch_epoch):.2f}% Label: {labels[0].cpu().numpy()} Pred: {predicted[0].cpu().numpy()}"
            )
            pbar.update()
        writer.add_scalar("Loss/train", np.mean(loss_batch_epoch), epoch)
        writer.add_scalar("Accuracy/train", np.mean(acc_batch_epoch), epoch)
        # torch.save(model.state_dict(), f"model_epochss.pth")

        pbar.close()
        model.eval()  # Put model in evaluation mode
        correct_val_predictions = 0
        total_val_samples = 0
        val_pbar = tqdm(
            total=len(val_dataloader),
            desc=f"Validating [{epoch+1}/{num_epochs}]",
            dynamic_ncols=True,
        )
        val_acc_batch_epoch = []
        val_loss_batch_epoch = []
        with torch.no_grad():  # No need to track gradients in validation
            for i, (video_val, audio, labels_val) in enumerate(val_dataloader):
                video_val = video_val.cuda()
                labels_val = labels_val.cuda()
                audio = audio.cuda()

                outputs_val = model(video_val, audio)
                loss = criterion(outputs_val, labels_val)
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val_samples += labels_val.size(0)
                correct_val_predictions += (predicted_val == labels_val).sum().item()

                avg_val_acc = (correct_val_predictions / total_val_samples) * 100

                val_acc_batch_epoch.append(avg_val_acc)
                val_loss_batch_epoch.append(loss.item())
                val_pbar.set_description(
                    f"Validating [{epoch+1}/{num_epochs}], Acc: {np.mean(val_acc_batch_epoch):.2f}%"
                )
                val_pbar.update()
        writer.add_scalar("Accuracy/val", np.mean(val_acc_batch_epoch), epoch)
        writer.add_scalar("Loss/val", np.mean(val_loss_batch_epoch), epoch)
        val_pbar.close()

        # Save the model if the validation accuracy has increased
        if np.mean(val_acc_batch_epoch) > best_val_accuracy:
            best_val_accuracy = np.mean(val_acc_batch_epoch)
            torch.save(model.state_dict(), os.path.join(work_dir, f"model_epochss.pth"))
            print(f"Model saved with validation accuracy: {best_val_accuracy:.2f}%")

# In[ ]: