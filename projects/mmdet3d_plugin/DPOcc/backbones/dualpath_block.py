import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule
from einops import rearrange
from .modules import BottleNeckASPP, SwinBlock


class VoxelToPillar(nn.Module):
    """Merge the four commented VoxelToPillar prototypes into one module.

    The original feature pipeline is retained: an MLP processes every voxel's
    height vector, min/max/mean statistics are extracted, each statistic is
    expanded, and a 1x1x1 convolution produces a pillar value.  The expansion
    and 1x1x1 convolution are evaluated in their algebraically fused form to
    avoid materialising a ``[B, 3I, C, X, Y]`` tensor.  This is mathematically
    equivalent and keeps gradients for both original layers.
    """

    def __init__(self, input_channels, intermediate_channels=64,
                 output_channels=1, final_kernel_size=None):
        super().__init__()
        self.input_channels = input_channels
        self.intermediate_channels = intermediate_channels
        self.output_channels = output_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, intermediate_channels),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(inplace=True),
            nn.Linear(intermediate_channels, intermediate_channels),
            nn.BatchNorm1d(intermediate_channels),
            nn.ReLU(inplace=True))
        self.feature_expand = nn.Linear(1, intermediate_channels)
        self.conv1x1 = nn.Conv3d(
            intermediate_channels * 3, output_channels, kernel_size=1)
        self.batch_norm = nn.BatchNorm3d(output_channels)
        self.final_conv = (nn.Conv3d(output_channels, 1,
                                    kernel_size=(1, 1, final_kernel_size))
                           if final_kernel_size is not None else None)

    def _expanded_conv(self, statistics):
        """Apply feature_expand + conv1x1 without the large expanded tensor."""
        expand_weight = self.feature_expand.weight[:, 0]
        expand_bias = self.feature_expand.bias
        conv_weight = self.conv1x1.weight.reshape(
            self.output_channels, 3, self.intermediate_channels)
        effective_weight = (
            conv_weight * expand_weight.view(1, 1, -1)).sum(-1)
        effective_bias = (
            conv_weight * expand_bias.view(1, 1, -1)).sum((1, 2))
        if self.conv1x1.bias is not None:
            effective_bias = effective_bias + self.conv1x1.bias
        output = torch.einsum('bchwk,ok->bochw', statistics,
                              effective_weight)
        return output + effective_bias.view(1, -1, 1, 1, 1)

    def forward(self, x):
        if x.ndim != 5:
            raise ValueError('VoxelToPillar expects [B, C, X, Y, Z], but got '
                             '{}'.format(tuple(x.shape)))
        if x.shape[-1] != self.input_channels:
            raise ValueError('Configured voxel height is {}, but input height '
                             'is {}'.format(self.input_channels, x.shape[-1]))

        batch_size, channels, height, width, features = x.shape
        mlp_output = self.mlp(x.reshape(-1, features))
        mlp_output = mlp_output.reshape(
            batch_size, channels, height, width,
            self.intermediate_channels)
        statistics = torch.stack(
            (mlp_output.min(dim=-1)[0], mlp_output.max(dim=-1)[0],
             mlp_output.mean(dim=-1)), dim=-1)
        pillar = self._expanded_conv(statistics)
        pillar = F.relu(self.batch_norm(pillar), inplace=True)
        if self.final_conv is not None:
            pillar = self.final_conv(pillar).squeeze(1)
            return F.adaptive_avg_pool3d(
                pillar, (channels, height, width))
        return pillar.squeeze(1)


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention used by the density BEV branch."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding)

    def forward(self, x):
        pooled = torch.cat(
            (x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True)[0]),
            dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class ChannelAttention(nn.Module):
    """Shared-MLP channel attention with safe support for small channels."""

    def __init__(self, channels, ratio=16):
        super().__init__()
        hidden_channels = max(channels // ratio, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False))

    def forward(self, x):
        avg_score = self.mlp(F.adaptive_avg_pool2d(x, 1))
        max_score = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg_score + max_score)


class DensityAwareBEV(nn.Module):
    """Density extraction followed by spatial and channel attention.

    This consolidates ``CombinedModel[1-3]`` from the commented experiment
    into one channel-agnostic module while retaining its operation order.
    """

    def __init__(self, channels, reduction=2, dilation=2):
        super().__init__()
        hidden_channels = max(channels // reduction, 1)
        self.density_extractor = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3,
                      padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True))
        self.spatial_attention = SpatialAttention()
        self.channel_attention = ChannelAttention(hidden_channels)
        self.restore_channels = nn.Conv2d(
            hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        enhanced = self.density_extractor(x)
        enhanced = self.spatial_attention(enhanced)
        enhanced = self.channel_attention(enhanced)
        return self.restore_channels(enhanced)
class DualpathTransformerBlock(BaseModule):
    def __init__(self,
                in_channels,
                channels,
                stride=1,
                norm_cfg=None,
                init_cfg=None,
                coeff_bias=True,
                aspp_drop=0.1,
                use_density_enhancement=True,
                voxel_height=None,
                pillar_intermediate_channels=64,
                first_pillar_stage=False,
                **kwargs):
        super().__init__(init_cfg=init_cfg)
        
        self.in_channels = in_channels
        self.channels = channels
        self.stride = stride
        self.norm_cfg = norm_cfg
        self.use_density_enhancement = use_density_enhancement
        if voxel_height is None:
            raise ValueError('voxel_height must be provided to '
                             'DualpathTransformerBlock')
        self.kwargs = kwargs
        self.shift = (self.kwargs['layer_index'] % 2) == 1
        
        self.multihead_base_channel = 32
        self.num_heads = int(self.channels / self.multihead_base_channel)
        
        # build skip connection
        if self.stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                build_norm_layer(norm_cfg, channels)[1])
        else:
            self.downsample = nn.Identity()
        
        self.input_conv = nn.Sequential(
            nn.Conv3d(in_channels, channels, kernel_size=3, 
                padding=1, stride=stride, bias=False),
            build_norm_layer(norm_cfg, channels)[1],
            nn.ReLU(inplace=True),
        )
        
        # shared window attention
        self.bev_encoder = SwinBlock(
            embed_dims=self.channels,
            num_heads=self.num_heads,
            feedforward_channels=self.channels,
            window_size=7,
            drop_path_rate=0.2,
            shift=self.shift)

        # Original statistical voxel-to-pillar and density-aware branches are
        # registered here (never constructed inside forward).
        self.voxel_to_pillar = VoxelToPillar(
            voxel_height,
            pillar_intermediate_channels,
            output_channels=voxel_height if first_pillar_stage else 1,
            final_kernel_size=(voxel_height if first_pillar_stage else None))
        self.density_enhancer = (DensityAwareBEV(self.channels)
                                 if use_density_enhancement else nn.Identity())
        
        # aspp in global path
        self.aspp = BottleNeckASPP(
            inplanes=self.channels,
            norm_cfg=self.norm_cfg,
            dropout=aspp_drop)
        
        # soft weights for fusion
        self.combine_coeff = nn.Conv3d(
            self.channels, 1, kernel_size=1, bias=coeff_bias)

    def forward(self, x):
        # Keep a reference for the residual; cloning this full 3D tensor
        # needlessly doubles peak activation memory.
        input_identity = x
        x = self.input_conv(x)
        x_bev = self.voxel_to_pillar(x)
        batch_size = x_bev.shape[0]
        
        x = rearrange(x, 'b c x y z -> (b z) c x y')
        x = torch.cat((x_bev, x), dim=0)
        x = self.bev_encoder(x)  # ReLU output
        x_bev, x = x[:batch_size], x[batch_size:]
        x = rearrange(x, '(b z) c x y -> b c x y z', b=batch_size)
        x_bev = self.density_enhancer(x_bev)
        x_bev = self.aspp(x_bev)
        coeff = self.combine_coeff(x).sigmoid()
        x = x + coeff * x_bev.unsqueeze(-1)
        return x + self.downsample(input_identity)
