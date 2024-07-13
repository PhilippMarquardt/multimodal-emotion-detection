#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial

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
import random

torch.manual_seed(42)


class CustomDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, skip_frames=1):
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
        labels = [self.labels[index]] * self.num_frames
        video, audio, info = read_video(video_path, pts_unit="sec")
        sample_rate = info.get("audio_fps")
        spectrogram_transform = MelSpectrogram(sample_rate=sample_rate)
        audio = audio.mean(dim=0)  # Convert stereo to mono if necessary
        audio_raw = audio.clone()  # Save a copy of raw audio

        # Sample frames
        total_frames = video.shape[0]
        selected_frames = []
        start_frame = 0

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
        splice_clip = random.random() < 0.5
        if splice_clip:
            # Choose a random clip to splice in
            splice_index = random.randint(0, len(self.file_paths) - 1)

            splice_video_path = self.file_paths[splice_index]
            # print(splice_video_path)
            splice_video, splice_audio, _ = read_video(
                splice_video_path, pts_unit="sec"
            )
            splice_audio = splice_audio.mean(
                dim=0
            )  # Convert stereo to mono if necessary
            if splice_audio.dim() == 1:
                splice_audio = splice_audio.unsqueeze(0)
            # Choose a random number of frames to replace at the end
            splice_frames = random.randint(1, self.num_frames - 4)
            # print(self.labels[splice_index])
            labels[-splice_frames:] = [self.labels[splice_index]] * splice_frames
            # splice_frames = self.num_frames // 2
            # Splice in the video frames

            selected_frames_second = []

            total_frames_second = splice_video.shape[0]
            # print("total frames second", total_frames_second)
            start_frame = torch.randint(
                0, total_frames_second - splice_frames * self.skip_frames + 1, ()
            )
            frame_indices = torch.arange(
                start_frame,
                start_frame + splice_frames * self.skip_frames,
                self.skip_frames,
            )
            selected_frames_second = splice_video[frame_indices]
            selected_frames = selected_frames[:-splice_frames]
            # print(selected_frames_second.shape)
            # print(selected_frames.shape)
            splice_video = splice_video[frame_indices]

            selected_frames = torch.cat((selected_frames, splice_video), dim=0)
            # selected_frames = selected_frames.squeeze(0)
            # print(selected_frames.shape)
            # Splice in the audio frames
            splice_audio_samples_per_frame = sample_rate / info["video_fps"]
            splice_audio_samples = int(splice_audio_samples_per_frame * splice_frames)

            # Select the corresponding audio samples from the second clip
            splice_audio_start_sample = int(
                splice_audio_samples_per_frame * start_frame
            )
            splice_audio_end_sample = splice_audio_start_sample + splice_audio_samples

            splice_audio_raw = splice_audio[
                :, splice_audio_start_sample:splice_audio_end_sample
            ]

            # Remove the corresponding samples from the end of the original audio
            audio_raw = audio_raw[:, :-splice_audio_samples]

            # Concatenate the selected audio from the second clip
            audio_raw = torch.cat((audio_raw, splice_audio_raw), dim=1)
        # Generate the MelSpectrogram from the selected raw audio
        spectrogram = spectrogram_transform(audio_raw)
        # resized_spectrogram = F.interpolate(spectrogram.unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
        # spectrogram = spectrogram.unsqueeze(0)  # Add a channel dimension
        spectrogram = (spectrogram - spectrogram.min()) / (
            spectrogram.max() - spectrogram.min()
        )  # Normalize to [0, 1]

        spectrogram = F.interpolate(
            spectrogram.unsqueeze(0),
            size=(512, 1024),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        # print("Spectogramshaoe", spectrogram.shape)
        #         spectrogram_resize_transform = Resize(size=(self.resize_longest, self.resize_longest))
        #         spectrogram = spectrogram_resize_transform(spectrogram)
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
        #    selected_frames[0 : self.num_frames, :, :]
        #    .permute(3, 0, 1, 2)
        #    .float(),
        #    spectrogram.repeat(3, 1, 1),
        #    audio_raw,
        #    sample_rate,
        #    labels,
        # )
        return (
            selected_frames.permute(3, 0, 1, 2).float(),
            spectrogram.repeat(3, 1, 1),
            torch.tensor(labels, dtype=torch.long),
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
        train_dataset, batch_size=8, shuffle=True, num_workers=2
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
    import random

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
            encoder_layers = TransformerEncoderLayer(d_model=512, nhead=4)
            self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=2)

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
            self.audio_model = models.resnet50(pretrained=True)
            num_ftrs_audio = self.audio_model.fc.in_features
            self.audio_model.fc = nn.Linear(num_ftrs_audio, 512)
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
                # print(frame.shape)
                frame_embedding = self.image_model(frame)
                frame_embeddings[:, i, :] = frame_embedding
            # print("done")
            # Pass frame embeddings through BiLSTM
            output, _ = self.bilstm(
                frame_embeddings
            )  # [batch_size, num_frames, hidden_size*2]

            # Apply FC1 and FC2 to each frame
            output = nn.functional.relu(
                self.fc1(output)
            )  # [batch_size, num_frames, 512]

            audio_outputs = self.audio_model(audio)

            # Concatenate features from the two streams
            # combined = torch.cat((video_outputs, audio_outputs), dim=1)
            output = self.fc2(output)  # [batch_size, num_frames, 8]

            return output

    class VideoAudioNetLSTM(nn.Module):
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
            self.audio_model = models.resnet50(pretrained=True)
            num_ftrs_audio = self.audio_model.fc.in_features
            self.audio_model.fc = nn.Linear(num_ftrs_audio, 512)
            # Linear layers
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(1024, 8)  # Assuming 8 classes

        def forward(self, video, audio):
            batch_size, c, num_frames, h, w = video.shape

            # Empty tensor to store the embeddings for each frame
            frame_embeddings = torch.empty(
                (batch_size, num_frames, 512), device=video.device
            )

            for i in range(num_frames):
                frame = video[:, :, i, :, :]
                # print(frame.shape)
                frame_embedding = self.image_model(frame)
                frame_embeddings[:, i, :] = frame_embedding
            # print("done")
            # Pass frame embeddings through BiLSTM
            output, _ = self.bilstm(
                frame_embeddings
            )  # [batch_size, num_frames, hidden_size*2]

            # Apply FC1 and FC2 to each frame
            output = nn.functional.relu(
                self.fc1(output)
            )  # [batch_size, num_frames, 512]

            audio_outputs = self.audio_model(audio)

            # Concatenate features from the two streams
            audio_outputs = audio_outputs.unsqueeze(1).expand(
                -1, num_frames, -1
            )  # [batch_size, num_frames, 512]

            # Concatenate the output and audio_outputs tensors along the last dimension
            output = torch.cat([output, audio_outputs], dim=-1)
            # print(output.shape)
            output = self.fc2(output)  # [batch_size, num_frames, 8]

            return output

    def _unsqueeze(x: torch.Tensor, target_dim: int, expand_dim: int):
        tensor_dim = x.dim()
        if tensor_dim == target_dim - 1:
            x = x.unsqueeze(expand_dim)
        elif tensor_dim != target_dim:
            raise ValueError(f"Unsupported input dimension {x.shape}")
        return x, tensor_dim

    class MViTSeq(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # Replace final linear layer for multi-label classification
            num_ftrs = base_model.head[1].in_features
            base_model.head[1] = nn.Linear(
                num_ftrs, 8
            )  # assuming num_classes is defined

        def forward(self, x):
            # Get embeddings before classification
            x = _unsqueeze(x, 5, 2)[0]
            x = self.base_model.conv_proj(x)
            x = x.flatten(2).transpose(1, 2)
            x = self.base_model.pos_encoding(x)
            block_outputs = []
            thw = (
                self.base_model.pos_encoding.temporal_size,
            ) + self.base_model.pos_encoding.spatial_size
            for block in self.base_model.blocks:
                x, thw = block(x, thw)
                block_outputs.append(x)
            x = self.base_model.norm(x)
            embeds = x[:, 0]

            # Apply final linear layer with sigmoid activation for multi-label classification
            multi_label_output = torch.sigmoid(self.base_model.head(embeds))

            return multi_label_output, block_outputs

    class AVNet(nn.Module):
        def __init__(self):
            super().__init__()
            # Pretrained models for video and audio processing
            self.video_model = models.video.mvit_v2_s(pretrained=True)
            self.video_model = MViTSeq(self.video_model)
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
            self.audio_model = models.resnet50(pretrained=True)

            # Replace the final fully connected layer of the audio model
            num_ftrs_audio = self.audio_model.fc.in_features
            self.audio_model.fc = nn.Linear(num_ftrs_audio, 512)
            print(self.video_model)
            # Replace the final layer of the video model
            # num_ftrs_video = self.video_model.head[1].in_features
            # self.video_model.head[1] = nn.Linear(num_ftrs_video, 512)
            # self.video_model.blocks[1] = nn.Identity()
            # MLP for combining features from the two streams
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 8)  # Assuming 8 classes

        def forward(self, video, audio_spectrogram):
            # Process video and audio
            # print(video.shape)
            video_outputs = self.video_model(video)
            print(video_outputs[-1].shape)
            # audio_outputs = self.audio_model(audio_spectrogram)

            # Concatenate features from the two streams
            # combined = torch.cat((video_outputs, audio_outputs), dim=1)

            # Pass the combined features through the MLP
            combined = F.relu(self.fc1(video_outputs))
            outputs = self.fc2(combined)

            return outputs

    model = VideoAudioNetLSTM()
    model.cuda()
    work_dir = "video_audio_lstm"
    os.mkdir(work_dir)

    def contains_invalid_values(tensor):
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    # model.load_state_dict(torch.load("model_epochss.pth"))
    num_epochs = 100  # Example number of epochs, adjust as needed
    criterion = torch.nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
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
        acc_per_batch = []
        loss_per_batch = []
        for i, (videos, audio, labels) in enumerate(dataloader):
            if contains_invalid_values(audio):
                # print("brokrne")
                continue
            videos = videos.cuda()
            audio = audio.cuda()
            labels = labels.cuda()

            outputs = model(videos, audio)  # Assuming model outputs raw logits
            loss = criterion(outputs.view(-1, 8), labels.view(-1))

            loss.backward()
            optimizer.step()
            model.zero_grad()

            # Compute accuracy
            _, predicted = torch.max(outputs, 2)
            correct_predictions = (predicted == labels).sum().item()
            total_samples = np.prod(labels.shape)

            # Calculate loss and accuracy for the current batch
            batch_loss = loss.item()
            batch_acc = (correct_predictions / total_samples) * 100

            acc_per_batch.append(batch_acc)
            loss_per_batch.append(batch_loss)

            # Update progress bar description
            pbar.set_description(
                f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(loss_per_batch):.4f}, Acc: {np.mean(acc_per_batch):.2f}% Label: {labels[0].cpu().numpy()} Pred: {predicted[0].cpu().numpy()}"
            )
            pbar.update()
        writer.add_scalar("Loss/train", np.mean(loss_per_batch), epoch)
        writer.add_scalar("Accuracy/train", np.mean(acc_per_batch), epoch)
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
        val_loss_per_batch = []
        val_acc_per_batch = []
        with torch.no_grad():  # No need to track gradients in validation
            for i, (video_val, audio_val, labels_val) in enumerate(val_dataloader):
                if contains_invalid_values(audio_val):

                    continue
                video_val = video_val.cuda()
                labels_val = labels_val.cuda()
                audio_val = audio_val.cuda()
                outputs_val = model(
                    video_val, audio_val
                )  # Assuming 2 is no longer needed

                # Compute loss
                loss_val = criterion(outputs_val.view(-1, 8), labels_val.view(-1))
                val_loss_per_batch.append(loss_val.item())

                # Compute accuracy
                _, predicted_val = torch.max(outputs_val, 2)
                correct_val_predictions = (predicted_val == labels_val).sum().item()
                total_val_samples = np.prod(labels_val.shape)

                # Calculate accuracy for the current batch
                batch_val_acc = (correct_val_predictions / total_val_samples) * 100
                val_acc_per_batch.append(batch_val_acc)
                val_pbar.set_description(
                    f"Validating [{epoch+1}/{num_epochs}], Acc: {np.mean(val_acc_per_batch):.2f}%"
                )
                val_pbar.update()
        writer.add_scalar("Accuracy/val", np.mean(val_acc_per_batch), epoch)

        val_pbar.close()

        # Save the model if the validation accuracy has increased
        if np.mean(val_acc_per_batch) > best_val_accuracy:
            best_val_accuracy = np.mean(val_acc_per_batch)
            torch.save(
                model.state_dict(), os.path.join(work_dir, f"lstmmultilabel.pth")
            )
            print(f"Model saved with validation accuracy: {best_val_accuracy:.2f}%")

# In[ ]: