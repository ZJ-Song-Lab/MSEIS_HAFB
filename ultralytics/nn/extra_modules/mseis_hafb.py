import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.jit import Final
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Optional, Dict, Union
from einops import rearrange, reduce
from collections import OrderedDict

from ..backbone.UniRepLKNet import get_bn, get_conv2d, NCHWtoNHWC, GRNwithNHWC, SEBlock, NHWCtoNCHW, fuse_bn, merge_dilated_into_large_kernel
from ..backbone.rmt import RetBlock, RelPos2d
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad, LightConv, ConvTranspose
from ..modules.block import get_activation, ConvNormLayer, BasicBlock, BottleNeck, RepC3, C3, C2f, Bottleneck
from .attention import *
from ..backbone.MambaOut import GatedCNNBlock_BCHW
from ultralytics.utils.torch_utils import fuse_conv_and_bn, make_divisible

from timm.layers import CondConv2d, DropPath, trunc_normal_, use_fused_attn, to_2tuple

__all__ = ['CSP_MutilScaleEdgeInformationEnhance', 'CSP_MultiScaleEdgeInformationSelect', 'EdgeEnhancer', 'MutilScaleEdgeInformationEnhance',
           'MultiScaleEdgeInformationSelection', 'MutilScaleEdgeInfoGenetator', 'HaarWaveletConv', 'HierarchicalAttentionFusionBlock',
           'ContrastDrivenFeatureAggregation', 'SobelConv' , 'ConvEdgeFusion'
           ]


class HierarchicalAttentionFusionBlock(nn.Module):
    # Hierarchical Attention Fusion Block (HAFB) for cross-scale feature integration
    # Key improvements: Parallel local-global attention branches and prompt-guided feature gating
    def __init__(self, input_dims, output_dim, group=False):
        super(HierarchicalAttentionFusionBlock, self).__init__()
        dim1, dim2 = input_dims
        hidden_dim = output_dim // 2

        # Feature projection layers for dimension alignment
        self.proj1 = Conv(dim1, hidden_dim, 1, act=False)  # Project higher-level feature
        self.proj2 = Conv(dim2, hidden_dim, 1, act=False)  # Project lower-level feature
        
        # Baseline fusion path for initial feature combination
        self.baseline_fusion = Conv(hidden_dim, output_dim, 3, g=4)  # 3x3 convolution

        # Multi-scale attention mechanisms
        # Local attention branch (small patch size) for fine-grained ship contours
        self.local_attn1 = LocalGlobalAttention(hidden_dim, patch_size=2)  # For higher-level feature
        self.local_attn2 = LocalGlobalAttention(hidden_dim, patch_size=2)  # For lower-level feature
        # Global attention branch (large patch size) for spatial context
        self.global_attn1 = LocalGlobalAttention(hidden_dim, patch_size=4)  # For higher-level feature
        self.global_attn2 = LocalGlobalAttention(hidden_dim, patch_size=4)  # For lower-level feature

        # Feature reorganization block
        self.dim_reduction = Conv(output_dim * 3, output_dim, 1)  # Channel squeeze
        self.refinement = RepConv(output_dim, output_dim, 3, g=(16 if group else 1))  # Feature refinement
        self.final_proj = Conv(output_dim, output_dim, 1)  # Final projection

    def forward(self, features):
        # Unpack input features
        higher_level_feat, lower_level_feat = features
        
        # Project features to common hidden dimension
        proj_higher = self.proj1(higher_level_feat)
        proj_lower = self.proj2(lower_level_feat)
        
        # Baseline fusion path
        baseline_feat = self.baseline_fusion(proj_higher + proj_lower)

        # Apply multi-scale attention
        # Local attention for detailed ship structures
        local_higher = self.local_attn1(proj_higher)
        local_lower = self.local_attn2(proj_lower)
        # Global attention for contextual information
        global_higher = self.global_attn1(proj_higher)
        global_lower = self.global_attn2(proj_lower)
        
        # Combine attention outputs
        attn_higher = torch.cat([local_higher, global_higher], dim=1)
        attn_lower = torch.cat([local_lower, global_lower], dim=1)

        # Final fusion of all feature paths
        merged_feat = torch.cat([attn_higher, attn_lower, baseline_feat], dim=1)
        
        # Reorganize and refine features
        output = self.dim_reduction(merged_feat)
        output = self.refinement(output)
        output = self.final_proj(output)
        
        return output

class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        edge = self.pool(x)
        edge = x - edge
        edge = self.out_conv(edge)
        return x + edge


class MutilScaleEdgeInformationEnhance(nn.Module):
    def __init__(self, inc, bins):
        super().__init__()

        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv(inc, inc // len(bins), 1),
                Conv(inc // len(bins), inc // len(bins), 3, g=inc // len(bins))
            ))
        self.ees = []
        for _ in bins:
            self.ees.append(EdgeEnhancer(inc // len(bins)))
        self.features = nn.ModuleList(self.features)
        self.ees = nn.ModuleList(self.ees)
        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        x_size = x.size()
        out = [self.local_conv(x)]
        for idx, f in enumerate(self.features):
            out.append(self.ees[idx](F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True)))
        return self.final_conv(torch.cat(out, 1))


class MultiScaleEdgeInformationSelection(nn.Module):
    # Multi-Scale Edge Information Selection (MSEIS) module
    # Key improvements: Multi-scale edge extraction and spatial-channel feature gating
    def __init__(self, in_channels, scale_bins):
        super(MultiScaleEdgeInformationSelection, self).__init__()
        self.in_channels = in_channels
        self.scale_bins = scale_bins
        self.num_scales = len(scale_bins)

        # Local context extraction branch
        self.local_context = Conv(in_channels, in_channels, 3)  # Capture immediate structural cues

        # Multi-scale edge processing paths
        self.scale_processors = nn.ModuleList()
        self.edge_enhancers = nn.ModuleList()
        
        for scale in scale_bins:
            # Scale-specific feature extraction
            scale_processor = nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),  # Downsample to scale-specific size
                Conv(in_channels, in_channels // self.num_scales, 1),  # Channel adjustment
                Conv(in_channels // self.num_scales, in_channels // self.num_scales, 3, 
                     g=in_channels // self.num_scales)  # Scale-specific convolution
            )
            self.scale_processors.append(scale_processor)
            
            # Edge Enhancement Unit (EEU) for each scale
            self.edge_enhancers.append(EdgeEnhancer(in_channels // self.num_scales))

        # Spatial-Channel Feature Gating (SCFG) for adaptive feature selection
        self.feature_gate = DualDomainSelectionMechanism(in_channels * 2)
        
        # Final feature fusion
        self.output_conv = Conv(in_channels * 2, in_channels, 1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        # Extract local structural context
        local_feat = self.local_context(x)
        
        # Process multi-scale edge information
        scale_features = []
        for scale_proc, enhancer in zip(self.scale_processors, self.edge_enhancers):
            # Extract scale-specific features
            scale_feat = scale_proc(x)
            # Enhance edge information
            enhanced_feat = enhancer(scale_feat)
            # Upsample to original resolution
            enhanced_feat = F.interpolate(enhanced_feat, size=(height, width), 
                                         mode='bilinear', align_corners=True)
            scale_features.append(enhanced_feat)
        
        # Combine local context and multi-scale edge features
        combined_feat = torch.cat([local_feat] + scale_features, dim=1)
        
        # Adaptive feature selection via SCFG
        gated_feat = self.feature_gate(combined_feat)
        
        # Final feature refinement
        output = self.output_conv(gated_feat)
        
        return output


class CSP_MutilScaleEdgeInformationEnhance(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MutilScaleEdgeInformationEnhance(self.c, [3, 6, 9, 12]) for _ in range(n))


class CSP_MultiScaleEdgeInformationSelect(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(MultiScaleEdgeInformationSelection(self.c, [3, 6, 9, 12]) for _ in range(n))

class HaarWaveletConv(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWaveletConv, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        # h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        # v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        # d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x):
        B, _, H, W = x.size()
        x = F.pad(x, [0, 1, 0, 1], value=0)
        out = F.conv2d(x, self.haar_weights, bias=None, stride=1, groups=self.in_channels) / 4.0
        out = out.reshape([B, self.in_channels, 4, H, W])
        out = torch.transpose(out, 1, 2)
        out = out.reshape([B, self.in_channels * 4, H, W])

        # a (approximation): 低频信息，图像的平滑部分，代表了图像的整体结构。
        # h (horizontal): 水平方向的高频信息，捕捉水平方向上的边缘或变化。
        # v (vertical): 垂直方向的高频信息，捕捉垂直方向上的边缘或变化。
        # d (diagonal): 对角线方向的高频信息，捕捉对角线方向上的边缘或纹理。
        a, h, v, d = out.chunk(4, 1)

        # 低频，高频
        return a, h + v + d


class ContrastDrivenFeatureAggregation(nn.Module):
    def __init__(self, dim, num_heads=8, kernel_size=3, padding=1, stride=1,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.head_dim = dim // num_heads

        self.scale = self.head_dim ** -0.5

        self.wavelet = HaarWaveletConv(dim)

        self.v = nn.Linear(dim, dim)
        self.attn_fg = nn.Linear(dim, kernel_size ** 4 * num_heads)
        self.attn_bg = nn.Linear(dim, kernel_size ** 4 * num_heads)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

        self.input_cbr = nn.Sequential(
            Conv(dim, dim, 3),
            Conv(dim, dim, 3),
        )
        self.output_cbr = nn.Sequential(
            Conv(dim, dim, 3),
            Conv(dim, dim, 3),
        )

    def forward(self, x):
        x = self.input_cbr(x)
        bg, fg = self.wavelet(x)

        x = x.permute(0, 2, 3, 1)
        fg = fg.permute(0, 2, 3, 1)
        bg = bg.permute(0, 2, 3, 1)

        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)

        v_unfolded = self.unfold(v).reshape(B, self.num_heads, self.head_dim,
                                            self.kernel_size * self.kernel_size,
                                            -1).permute(0, 1, 4, 3, 2)
        attn_fg = self.compute_attention(fg, B, H, W, C, 'fg')

        x_weighted_fg = self.apply_attention(attn_fg, v_unfolded, B, H, W, C)

        v_unfolded_bg = self.unfold(x_weighted_fg.permute(0, 3, 1, 2)).reshape(B, self.num_heads, self.head_dim,
                                                                               self.kernel_size * self.kernel_size,
                                                                               -1).permute(0, 1, 4, 3, 2)
        attn_bg = self.compute_attention(bg, B, H, W, C, 'bg')

        x_weighted_bg = self.apply_attention(attn_bg, v_unfolded_bg, B, H, W, C)

        x_weighted_bg = x_weighted_bg.permute(0, 3, 1, 2)

        out = self.output_cbr(x_weighted_bg)

        return out

    def compute_attention(self, feature_map, B, H, W, C, feature_type):
        attn_layer = self.attn_fg if feature_type == 'fg' else self.attn_bg
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        feature_map_pooled = self.pool(feature_map.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        attn = attn_layer(feature_map_pooled).reshape(B, h * w, self.num_heads,
                                                      self.kernel_size * self.kernel_size,
                                                      self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        return attn

    def apply_attention(self, attn, v, B, H, W, C):
        x_weighted = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, self.dim * self.kernel_size * self.kernel_size, -1)
        x_weighted = F.fold(x_weighted, output_size=(H, W), kernel_size=self.kernel_size,
                            padding=self.padding, stride=self.stride)
        x_weighted = self.proj(x_weighted.permute(0, 2, 3, 1))
        x_weighted = self.proj_drop(x_weighted)
        return x_weighted


class SobelConv(nn.Module):
    def __init__(self, channel) -> None:
        super().__init__()

        sobel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        sobel_kernel_y = torch.tensor(sobel, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)
        sobel_kernel_x = torch.tensor(sobel.T, dtype=torch.float32).unsqueeze(0).expand(channel, 1, 1, 3, 3)

        self.sobel_kernel_x_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)
        self.sobel_kernel_y_conv3d = nn.Conv3d(channel, channel, kernel_size=3, padding=1, groups=channel, bias=False)

        self.sobel_kernel_x_conv3d.weight.data = sobel_kernel_x.clone()
        self.sobel_kernel_y_conv3d.weight.data = sobel_kernel_y.clone()

        self.sobel_kernel_x_conv3d.requires_grad = False
        self.sobel_kernel_y_conv3d.requires_grad = False

    def forward(self, x):
        return (self.sobel_kernel_x_conv3d(x[:, :, None, :, :]) + self.sobel_kernel_y_conv3d(x[:, :, None, :, :]))[:, :,
               0]


class MutilScaleEdgeInfoGenetator(nn.Module):
    def __init__(self, inc, oucs) -> None:
        super().__init__()

        self.sc = SobelConv(inc)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_1x1s = nn.ModuleList(Conv(inc, ouc, 1) for ouc in oucs)

    def forward(self, x):
        outputs = [self.sc(x)]
        outputs.extend(self.maxpool(outputs[-1]) for _ in self.conv_1x1s)
        outputs = outputs[1:]
        for i in range(len(self.conv_1x1s)):
            outputs[i] = self.conv_1x1s[i](outputs[i])
        return outputs


class ConvEdgeFusion(nn.Module):
    def __init__(self, inc, ouc) -> None:
        super().__init__()

        self.conv_channel_fusion = Conv(sum(inc), ouc // 2, k=1)
        self.conv_3x3_feature_extract = Conv(ouc // 2, ouc // 2, 3)
        self.conv_1x1 = Conv(ouc // 2, ouc, 1)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.conv_1x1(self.conv_3x3_feature_extract(self.conv_channel_fusion(x)))
        return x
