import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PConv(nn.Module):
    """Partial Convolution (CVPR 2023) — only apply conv on a subset of channels."""

    def __init__(self, dim, n_div=4, forward="split_cat"):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size=7, stride=1, padding=3, bias=False)
        self.forward_type = forward

    def forward(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat((x1, x2), 1)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block (CVPR 2018)"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NexNetBlock(nn.Module):
    """NexNet Block: DW-Conv -> BatchNorm -> PW-Expansion -> GELU -> PW-Projection -> SE Attention"""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.spatial_mix = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 3 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(3 * dim, dim, kernel_size=1)
        self.attn = SEBlock(dim, reduction=4)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim, 1, 1)),
                                   requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        inp = x
        x = self.spatial_mix(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = self.attn(x)
        x = inp + self.drop_path(x)
        return x


class NexNet(nn.Module):
    def __init__(self, embedding_size=512, depths=None, dims=None):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if dims is None:
            dims = [48, 96, 192, 384]

        self.downsample_layers = nn.ModuleList()

        # Stem: 112 -> 56
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        # Stage 2-4 Downsamples
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        for i in range(4):
            stage = nn.Sequential(
                *[NexNetBlock(dim=dims[i]) for _ in range(depths[i])]
            )
            self.stages.append(stage)

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)

        # GDC (Global Depthwise Conv) head
        self.gdc = nn.Conv2d(dims[-1], dims[-1], kernel_size=7, groups=dims[-1], bias=False)
        self.bn_gdc = nn.BatchNorm2d(dims[-1])
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(dims[-1], embedding_size, bias=True)
        self.bn_fc = nn.BatchNorm1d(embedding_size)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        x = self.downsample_layers[2](x)
        x = self.stages[2](x)

        x = self.downsample_layers[3](x)
        x = self.stages[3](x)

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)

        x = self.gdc(x)
        x = self.bn_gdc(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.bn_fc(x)
        return x


def nexnet(embedding_size=512):
    return NexNet(embedding_size=embedding_size, depths=[2, 2, 6, 2], dims=[48, 96, 192, 384])

