""" Written by Mingyu """
import logging
from copy import deepcopy
import itertools
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import build_model_with_cfg, overlay_external_default_cfg
from .layers import DropPath, to_2tuple, trunc_normal_
from .registry import register_model
from .vision_transformer import checkpoint_filter_fn, _init_vit_weights

_logger = logging.getLogger(__name__)

# python3 -u -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=10071 train.py data/dataset/imagenet-mini --model DaViT_tiny --batch-size 8 --lr 1e-3 --native-amp --clip-grad 1.0 --output output/

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embeds[0].proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'DaViT_224': _cfg(),
    'DaViT_384': _cfg(input_size=(3, 384, 384), crop_pct=1.0),
    'DaViT_384_22k': _cfg(input_size=(3, 384, 384), crop_pct=1.0, num_classes=21841)
}


def _init_conv_weights(m):
    """ Weight initialization for Vision Transformers.
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=0.02)
        for name, _ in m.named_parameters():
            if name in ['bias']:
                nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0)


class MySequential(nn.Sequential):
    """ Multiple input/output Sequential Module.
    """
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs

######################################################################
## MLP(used in Chann and Spatial Block)
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


######################################################################
## Positional Encoding(in Chan, Spatial Attn)
class ConvPosEnc(nn.Module):
    """Depth-wise convolution to get the positional information.
    """
    def __init__(self, dim, k=3):
        super(ConvPosEnc, self).__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)

    def forward(self, x, size: Tuple[int, int]):
        B, N, C = x.shape
        H, W = size
        assert N == H * W

        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        feat = feat.flatten(2).transpose(1, 2)
        x = x + feat
        return x


######################################################################
## PatchEmbedding
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            embed_dim=96,
            overlapped=False):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        if patch_size[0] == 4:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(7, 7),
                stride=patch_size,
                padding=(3, 3))
            self.norm = nn.LayerNorm(embed_dim)
        if patch_size[0] == 2:
            kernel = 3 if overlapped else 2
            pad = 1 if overlapped else 0
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=to_2tuple(kernel),
                stride=patch_size,
                padding=to_2tuple(pad))
            self.norm = nn.LayerNorm(in_chans)

    def forward(self, x, size):
        H, W = size
        dim = len(x.shape)
        if dim == 3:
            B, HW, C = x.shape
            x = self.norm(x)
            x = x.reshape(B,
                          H,
                          W,
                          C).permute(0, 3, 1, 2).contiguous()

        B, C, H, W = x.shape
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)
        newsize = (x.size(2), x.size(3))
        x = x.flatten(2).transpose(1, 2)
        if dim == 4:
            x = self.norm(x)
        return x, newsize


#########################################################################
## Channel Block & Attention
class ChannelAttention(nn.Module):
    r""" Channel based self attention.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of the groups.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k = k * self.scale
        attention = k.transpose(-1, -2) @ v
        attention = attention.softmax(dim=-1)
        x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class ChannelBlock(nn.Module):
    r""" Channel-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, layer_id=0, layer_offset_id=0):
        super().__init__()
        '''
        self.attn : ChannelAttention(
                        (qkv): Linear(in_features=96, out_features=288, bias=True)
                        (proj): Linear(in_features=96, out_features=96, bias=True)
                    )
        self.mlp : Mlp(
                        (fc1): Linear(in_features=96, out_features=384, bias=True)
                        (act): GELU()
                        (fc2): Linear(in_features=384, out_features=96, bias=True)
                    )
        self.norm : LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        '''
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.attn = ChannelAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ffn : Feed-Foward Network
        # if self.ffn:
        #     self.norm2 = norm_layer(dim)
        #     mlp_hidden_dim = int(dim * mlp_ratio)
        #     self.mlp = Mlp(
        #         in_features=dim,
        #         hidden_features=mlp_hidden_dim,
        #         act_layer=act_layer)

    def forward(self, x, size):
        # import pdb;pdb.set_trace()
        '''
        input x : [B, N, C](ex. [1, 3136, 96])
        size : int(sqrt(N)), ex:(tensor(56), tensor(56))

        '''
        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        cur = self.attn(cur)
        x = x + self.drop_path(cur)

        # x = self.cpe[1](x, size)
        # if self.ffn:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


#########################################################################
## Window Size Processing
def window_partition(x, window_size: int):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


#########################################################################
## Spatial Block and Attention
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        return x


class SpatialBlock(nn.Module):
    r""" Spatial-wise Local Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        ffn (bool): If False, pure attention network without FFNs
    """

    def __init__(self, dim, num_heads, window_size=7,
                 mlp_ratio=4., qkv_bias=True, drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 ffn=True, layer_id=0, layer_offset_id=0):
        super().__init__()
        self.dim = dim
        self.ffn = ffn
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # if self.ffn:
        #     self.norm2 = norm_layer(dim)
        #     mlp_hidden_dim = int(dim * mlp_ratio)
        #     self.mlp = Mlp(
        #         in_features=dim,
        #         hidden_features=mlp_hidden_dim,
        #         act_layer=act_layer)

    def forward(self, x, size):
        '''
        1) input x에 바로 Positional Encodding 삽입
        2) padding으로 윈도우에 비례하게 사이즈 맞춤
        3) window_partition
        4) attention 후 resize
        5) 패딩 제거
        6) 다시 Positional Encoding 처리 및 FFN
        '''
        # import pdb;pdb.set_trace()
        H, W = size
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # 1)
        shortcut = self.cpe[0](x, size)
        x = self.norm1(shortcut)
        x = x.view(B, H, W, C)

        # 2)
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # 3)
        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # 4)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1,
                                         self.window_size,
                                         self.window_size,
                                         C)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # 5)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # 6)
        # x = self.cpe[1](x, size)
        # if self.ffn:
        #     x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, size


#########################################################################
## Fusion Block
class Fusion(nn.Module):      
    '''
    96 또는 64의 concat된 채널이 오면, ln + mlp + drop_path 등 진행
    '''
    def __init__(self, dim, mlp_ratio=4., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, ffn=True):
        super().__init__()

        self.cpe = ConvPosEnc(dim=dim, k=3)
        self.ffn = ffn
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # ffn: Feed-Forward Network
        if self.ffn:
            self.norm = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, x, size):
        x = self.cpe(x, size)
        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm(x)))
        
        return x, size

#########################################################################
## DaViT
class DaViT(nn.Module):
    r""" Dual-Attention ViT

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dims (tuple(int)): Patch embedding dimension. Default: (64, 128, 192, 256)
        num_heads (tuple(int)): Number of attention heads in different layers. Default: (4, 8, 12, 16)
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        attention_types (tuple(str)): Dual attention types.
        ffn (bool): If False, pure attention network without FFNs
        overlapped_patch (bool): If True, use overlapped patch division during patch merging.
    """

    def __init__(self, in_chans=3, num_classes=1000, depths=(1, 1, 3, 1), patch_size=4,
                 embed_dims=(64, 128, 192, 256), num_heads=(3, 6, 12, 24), window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop_path_rate=0.1, norm_layer=nn.LayerNorm, attention_types=('spatial', 'channel'),
                 ffn=True, overlapped_patch=False, weight_init='',
                 img_size=224, drop_rate=0., attn_drop_rate=0.
                 ):
        super().__init__()
        '''
        feature -- 1/2로 나눠, spatial_attn -- concat -- LN -- mlp
                |- 1/2로 나눠, channel_attn -|

        embed_dims = (96, 192, 384, 768)
        num_heads = (2, 4, 8, 16)
        or
        embed_dims = (64, 128, 192, 256)
        num_heads = (1, 2, 4, 8)

        dpr을 1/2로 만들어야 함.
        '''

        architecture = [[index] * item for index, item in enumerate(depths)]
        self.architecture = architecture
        self.num_classes = num_classes

        # flops : 2312185856
        # param : 15401064
        num_heads = (2, 4, 8, 16)

        self.embed_dims = [embed_dims[i] // 2 for i in range(len(embed_dims))]
        # self.embed_dims = embed_dims

        self.num_heads = num_heads
        self.num_stages = len(self.embed_dims)

        # dpr을 1/2 size로 만들어야 됨.
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 2 * len(list(itertools.chain(*self.architecture))))]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, len(list(itertools.chain(*self.architecture))))]
        assert self.num_stages == len(self.num_heads) == (sorted(list(itertools.chain(*self.architecture)))[-1] + 1)

        self.img_size = img_size

        '''
        patch_embed && downsample
        chan : 3 -> [96, 192, 384, 768]
        conv2d : (7,7) -> (2,2)
        '''
        self.patch_embeds = nn.ModuleList([
            PatchEmbed(patch_size=patch_size if i == 0 else 2,
                       in_chans=in_chans if i == 0 else embed_dims[i - 1],  # chans : //2 버전이 아닌, 그냥 버전 사용
                       embed_dim=embed_dims[i],
                       overlapped=overlapped_patch)
            for i in range(self.num_stages)])

        # architecture : [[0], [1], [2, 2, 2], [3]]
        #                block_id, block_param : stage, 각 stage의 depth만큼 stage의 idx 저장
        #                layer_id, item : depth(몇 번째 depth인지), stage의 idx
        # attention_types : ['spatial', 'channel']
        # num_heads : [3, 6, 12, 24]
        # embed_dims : [96, 192, 384, 768]
        # mlp_ratio : 4.0
        # ffn : True
        '''
        총 4개의 stage(=architecture) 존재, 각 stage당 1개의 ModuleList(=N개의 depth) 존재
        각 stage 당 [1, 2, 3, 1]개의 depth 존재
        각 depth당 각 1개의 channel, spatial attn 존재
        
        merge and mlp : attention_type에 따른 drop_path 조정이 필요 없음.
        '''
        main_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            # 이전 layer의 depth의 수(=현 layer의 첫 depth의 시작점)
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))  # << 이 부분 고쳐야 됨
        
            block = nn.ModuleList([
                MySequential(*[
                    ChannelBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[(layer_id + layer_offset_id)],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        layer_id=layer_id,
                        layer_offset_id=layer_offset_id,
                    ) if attention_type == 'channel' else
                    SpatialBlock(
                        dim=self.embed_dims[item],
                        num_heads=self.num_heads[item],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        drop_path=dpr[(layer_id + layer_offset_id)],
                        norm_layer=nn.LayerNorm,
                        ffn=ffn,
                        window_size=window_size,
                        layer_id=layer_id,
                        layer_offset_id=layer_offset_id,
                    ) if attention_type == 'spatial' else None
                    for attention_id, attention_type in enumerate(attention_types)]
                ) for layer_id, item in enumerate(block_param)
            ])
            main_blocks.append(block)

        self.main_blocks = nn.ModuleList(main_blocks)

        # after all blocks
        self.norms = norm_layer(embed_dims[-1]) # chans : //2 버전이 아닌, 그냥 버전 사용
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[-1], num_classes)  # chans : //2 버전이 아닌, 그냥 버전 사용

        # merge and mlp
        fusion_blocks = []
        for block_id, block_param in enumerate(self.architecture):
            layer_offset_id = len(list(itertools.chain(*self.architecture[:block_id])))

            fusion_block = nn.ModuleList([
                MySequential(*[
                    Fusion(dim=embed_dims[item],
                           mlp_ratio=mlp_ratio,
                           drop_path=dpr[layer_id + layer_offset_id],
                           norm_layer=nn.LayerNorm,
                           ffn=ffn)]
                ) for layer_id, item in enumerate(block_param)
            ])
            
            fusion_blocks.append(fusion_block)
        self.fusion_blocks = nn.ModuleList(fusion_blocks)

        if weight_init == 'conv':
            self.apply(_init_conv_weights)
        else:
            self.apply(_init_vit_weights)

    def forward(self, x):
        '''
        개선안 : 동시에 진행하고 mlp 후 concat
        '''
        x, size = self.patch_embeds[0](x, (x.size(2), x.size(3)))
        features = [x]
        sizes = [size]
        branches = [0]

        # architecture : [[0], [1], [2, 2, 2], [3]]
        # attention_types : ['spatial', 'channel']
        # num_heads : [3, 6, 12, 24]
        # embed_dims : [96, 192, 384, 768]
        # mlp_ratio : 4.0
        # ---------------------------------self attn stage ---------------------------------------------- # 
        # 총 stage만큼 실행
        for block_index, block_param in enumerate(self.architecture):
            branch_ids = sorted(set(block_param))
            # branch_id가 0이 아닐 때 실행, 다운샘플링 실행
            # x, size : 이전 stage의 output(feature의 chan과 size)
            for branch_id in branch_ids:
                if branch_id not in branches:
                    x, size = self.patch_embeds[branch_id](features[-1], sizes[-1])
                    features.append(x)
                    sizes.append(size)
                    branches.append(branch_id)

            # 각 stage의 depth만큼 반복
            # features[branch_id], sizes[branch_id] : 현재 feature, 그 feature의 spatial_size 
            # block_index, layer_index, branch_id : 현재 stage, depth, stage_idx 
            # -> 즉 (0,0),(0,1),(0,2),(1,2),(2,2),(0,3)
            # features[branch_id] : 현 stage의 feature output 저장
            for layer_index, branch_id in enumerate(block_param):
                _, _, C = features[branch_id].shape
                x1, x2 = features[branch_id].split(C // 2, dim=-1)

                x1, _ = self.main_blocks[block_index][layer_index][0](x1, sizes[branch_id])
                x2, _ = self.main_blocks[block_index][layer_index][1](x2, sizes[branch_id])
                
                x = torch.cat([x1, x2], dim=-1)

                features[branch_id], _ = self.fusion_blocks[block_index][layer_index](x, sizes[branch_id])
        # ---------------------------------self attn stage ---------------------------------------------- # 

        features[-1] = self.avgpool(features[-1].transpose(1, 2))
        features[-1] = torch.flatten(features[-1], 1)
        x = self.norms(features[-1])
        x = self.head(x)
        return x


def _create_transformer(
        variant,
        pretrained=False,
        default_cfg=None,
        **kwargs):
    if default_cfg is None:
        default_cfg = deepcopy(default_cfgs[variant])
    overlay_external_default_cfg(default_cfg, kwargs)
    default_num_classes = default_cfg['num_classes']
    default_img_size = default_cfg['input_size'][-2:]

    num_classes = kwargs.pop('num_classes', default_num_classes)
    img_size = kwargs.pop('img_size', default_img_size)
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    model = build_model_with_cfg(
        DaViT, variant, pretrained,
        default_cfg=default_cfg,
        img_size=img_size,
        num_classes=num_classes,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)

    return model


@register_model
def DaViT_tiny(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24),
        depths=(1, 1, 3, 1), mlp_ratio=4., overlapped_patch=False, **kwargs)
    print(model_kwargs)
    return _create_transformer('DaViT_224', pretrained=pretrained, **model_kwargs)
# FLOPs: 4540244736, params: 28360168


@register_model
def DaViT_small(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dims=(96, 192, 384, 768), num_heads=(3, 6, 12, 24),
        depths=(1, 1, 9, 1), mlp_ratio=4., overlapped_patch=False, **kwargs)
    print(model_kwargs)
    return _create_transformer('DaViT_224', pretrained=pretrained, **model_kwargs)
# FLOPs: 8800488192, params: 49745896


@register_model
def DaViT_base(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=7, embed_dims=(128, 256, 512, 1024), num_heads=(4, 8, 16, 32),
        depths=(1, 1, 9, 1), mlp_ratio=4., overlapped_patch=False, **kwargs)
    print(model_kwargs)
    return _create_transformer('DaViT_224', pretrained=pretrained, **model_kwargs)
# FLOPs: 15510430720, params: 87954408


@register_model
def DaViT_large_window12_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=4, window_size=12, embed_dims=(192, 384, 768, 1536), num_heads=(6, 12, 24, 48),
        depths=(1, 1, 9, 1), mlp_ratio=4., overlapped_patch=False, **kwargs)
    print(model_kwargs)
    return _create_transformer('DaViT_384', pretrained=pretrained, **model_kwargs)
# FLOPs: 102966676992 (384x384), params: 196811752
