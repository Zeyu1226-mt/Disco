# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Copyright 2020 Ross Wightman
# Modified Model definition

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment
from einops import reduce
from .vit_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .helpers import load_pretrained
from .vit_utils import DropPath, to_2tuple, trunc_normal_

from .build import MODEL_REGISTRY
from torch import einsum, no_grad
from einops import rearrange, reduce, repeat
from Stage_one.Playeraction.network.TimeSformer.timesformer.models.agg_block.agg_block import AggregationBlock



def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # revise by xi, for input is [b,t,3,h,w]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        ##############
        # following is ori code
        B, C, T, H, W = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        return x[:, 0] # global feature
        #return x[:, 1:]#, x[:, 0]  # full feature and global

    def forward(self, x):
        x_f = self.forward_features(x)
        #x_f, x_cls = self.forward_features(x)
        #print("full: ", x_f.shape)  # forward_features:  torch.Size([B, 768])
        #print("cls: ", x_cls.shape)
        x_s = self.head(x_f)
        #print("head: ", x.shape)  # head:  torch.Size([B, 321])
        return x_f

class VisionTransformer_slot(nn.Module):
    """ Vision Transformere
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, num_action=4, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.):
        super().__init__()
        self.attention_type = attention_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        #print("input: ", x.shape)  # input:  torch.Size([B, 20, 3, 224, 224])
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            # x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        #return x[:, 0] # global feature
        return x[:, 1:], x[:, 0]  # full feature and global

    def forward(self, x):
        x_full, x_global = self.forward_features(x)
        #x_f, x_cls = self.forward_features(x)
        # print("full: ", x_full.shape)  # full:  torch.Size([4, 3920, 768]): T=20, N*N=196, N=14
        # print("cls: ", x_global.shape) # cls:  torch.Size([4, 768])
        #x_s = self.head(x_f)
        #print("head: ", x.shape)  # head:  torch.Size([B, 321])
        return x_full, x_global

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

@MODEL_REGISTRY.register()
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=cfg.DATA.TRAIN_CROP_SIZE, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

    def forward(self, x):
        x = self.model(x)
        return x

@MODEL_REGISTRY.register()
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)
    def forward(self, x):
        x = self.model(x)
        return x





@MODEL_REGISTRY.register()
class TimeSformerModified(nn.Module):
    def __init__(self, original_model, num_classes_identity, num_classes_action):
        super(TimeSformerModified, self).__init__()
        # 复制原始 TimeSformer 模型的前四个Block
        self.shared_blocks = nn.Sequential(*list(original_model.model.blocks[:6]))  # 前四个Block作为共享特征提取层

        # 分别定义两个分支，从原始模型中提取剩余Block
        self.identity_branch = nn.Sequential(*list(original_model.model.blocks[6:]))  # 用于身份预测的分支
        self.action_branch = nn.Sequential(*list(original_model.model.blocks[6:]))  # 用于动作预测的分支

        # 新定义的分类头
        self.identity_head = nn.Linear(original_model.model.embed_dim, num_classes_identity)
        self.action_head = nn.Linear(original_model.model.embed_dim, num_classes_action)

        self.attention_type = original_model.model.attention_type
        self.depth = original_model.model.depth
        self.dropout = nn.Dropout(0.)
        self.num_classes = original_model.model.num_classes
        self.num_features = self.embed_dim = original_model.model.embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=224, patch_size=16, in_chans=3, embed_dim=original_model.model.embed_dim)
        num_patches = self.patch_embed.num_patches

        ## Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, original_model.model.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, original_model.model.embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, 20, original_model.model.embed_dim))
            self.time_drop = nn.Dropout(p=0.)

        ## Attention Blocks
        #dpr = [x.item() for x in torch.linspace(0, 0.1, self.depth)]  # stochastic depth decay rule
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=original_model.model.embed_dim, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
        #         drop=0., attn_drop=0.1, drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #         attention_type=self.attention_type)
        #     for i in range(self.depth)])
        self.norm1 = nn.LayerNorm(original_model.model.embed_dim)
        self.norm2 = nn.LayerNorm(original_model.model.embed_dim)

        # Classifier head
        #self.head = nn.Linear(original_model.model.embed_dim, original_model.model.num_classes) if original_model.model.num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        # if self.attention_type == 'divided_space_time':
        #     i = 0
        #     for m in self.blocks.modules():
        #         m_str = str(m)
        #         if 'Block' in m_str:
        #             if i > 0:
        #                 nn.init.constant_(m.temporal_fc.weight, 0)
        #                 nn.init.constant_(m.temporal_fc.bias, 0)
        #             i += 1

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def feature_forward(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
        return x, B, T, W
    def forward(self, x):
        #print("x: ", x.shape)
        with no_grad():
            x, B, T, W = self.feature_forward(x)
            #print("x, B, T, W: ", x, B, T, W)
            # 通过共享的特征提取层
            for share_blk in self.shared_blocks:
                x = share_blk(x, B, T, W)
    #shared_features = self.shared_blocks(x, B, T, W)

        x1 = x
        x2 = x
        # 分别通过两个分支
        for identity_blk in self.identity_branch:
            x1 = identity_blk(x1, B, T, W)
        #identity_features = self.identity_branch(shared_features)
        #print("identity_features: ", x1.shape)
        x1_norm = self.norm1(x1)
        #print("identity: ", x1.shape)


        for action_blk in self.action_branch:
            x2 = action_blk(x2, B, T, W)
        #action_features = self.action_branch(shared_features)
        #print("action_features: ", x2.shape)
        x2_norm = self.norm2(x2)
        #print("action: ", x2.shape)

        player_scores = self.identity_head(x1_norm[:, 0])
        #print("player_scores: ", player_scores.shape)
        action_scores = self.action_head(x2_norm[:, 0])
        #print("action_scores: ", action_scores.shape)
        # # 平均池化 + 分类预测
        # identity_logits = self.identity_head(identity_features.mean(dim=1))  # 平均池化得到身份预测
        # action_logits = self.action_head(action_features.mean(dim=1))  # 平均池化得到动作预测

        return player_scores, x1_norm[:, 0], action_scores, x2_norm[:, 0]


# import torch
# import torch.nn as nn
# from functools import partial
# from timm.models.vision_transformer import Block, PatchEmbed
# from timm.models.layers import trunc_normal_

@MODEL_REGISTRY.register()
class TimeSformer_IA_pretrain(nn.Module):
    def __init__(self, original_model, num_classes_identity=321, num_classes_action=4, split_layer=6):
        super(TimeSformer_IA_pretrain, self).__init__()

        # 1. 加载共享组件
        self.patch_embed = original_model.model.patch_embed
        self.pos_drop = original_model.model.pos_drop
        self.time_drop = original_model.model.time_drop if hasattr(original_model.model, 'time_drop') else None
        self.cls_token = original_model.model.cls_token
        self.pos_embed = original_model.model.pos_embed
        self.attention_type = original_model.model.attention_type
        self.dropout = nn.Dropout(0.)
        if hasattr(original_model.model, 'time_embed'):
            self.time_embed = original_model.model.time_embed

        # 2. 共享层 (前split_layer层)
        self.shared_blocks = nn.ModuleList(list(original_model.model.blocks[:split_layer]))

        # 3. 身份分支 (split_layer层之后)
        self.identity_blocks = nn.ModuleList(list(original_model.model.blocks[split_layer:]))
        self.identity_norm = original_model.model.norm
        self.identity_head = nn.Linear(original_model.model.embed_dim, num_classes_identity)

        # 4. 动作分支 (split_layer层之后)
        self.action_blocks = nn.ModuleList(list(original_model.model.blocks[split_layer:]))
        self.action_norm = original_model.model.norm
        self.action_head = nn.Linear(original_model.model.embed_dim, num_classes_action)

        # 5. 初始化新添加的层
        self._init_weights_for_new_layers()

        # 6. 加载预训练权重
        self.load_pretrained_weights(original_model, split_layer)

    def _init_weights_for_new_layers(self):
        # 初始化分类头
        nn.init.normal_(self.identity_head.weight, std=0.02)
        nn.init.constant_(self.identity_head.bias, 0)
        nn.init.normal_(self.action_head.weight, std=0.02)
        nn.init.constant_(self.action_head.bias, 0)

    def load_pretrained_weights(self, original_model, split_layer):
        # 获取原始模型的预训练权重
        pretrained_dict = original_model.state_dict()
        model_dict = self.state_dict()

        # 1. 加载共享组件权重
        shared_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias',
                       'pos_embed', 'cls_token']
        if hasattr(original_model.model, 'time_embed'):
            shared_keys.append('time_embed')

        for key in shared_keys:
            if f'model.{key}' in pretrained_dict and key in model_dict:
                #print("shared")
                #print(pretrained_dict[f'model.{key}'])
                model_dict[key] = pretrained_dict[f'model.{key}']

        # 2. 加载共享层权重
        for i in range(split_layer):
            for key in pretrained_dict:
                if f'model.blocks.{i}.' in key:
                    #print("shared weights")
                    new_key = key.replace(f'model.blocks.{i}.', f'shared_blocks.{i}.')
                    if new_key in model_dict:
                        # print("share")
                        model_dict[new_key] = pretrained_dict[key]
        # print("len: ", len(original_model.model.blocks)) # 12
        # 3. 加载身份分支权重
        for i in range(split_layer, len(original_model.model.blocks)):
            for key in pretrained_dict:
                if f'model.blocks.{i}.' in key:
                    #print("identity weights")
                    #print(f'identity_blocks.{i - split_layer}.')
                    new_key = key.replace(f'model.blocks.{i}.', f'identity_blocks.{i - split_layer}.')
                    if new_key in model_dict:
                        #print("identity")
                        model_dict[new_key] = pretrained_dict[key]

        # 4. 加载动作分支权重 (与身份分支相同初始化)
        for i in range(split_layer, len(original_model.model.blocks)):
            #print("i: ", i)
            for key in pretrained_dict:
                #print("key: ", key)
                if f'model.blocks.{i}.' in key:
                    #ori_key = f'model.blocks.{i}.'
                    #print(f'action_blocks.{i - split_layer}.')
                    new_key = key.replace(f'model.blocks.{i}.', f'action_blocks.{i - split_layer}.')
                    #print("new: ", new_key)
                    if new_key in model_dict:
                        #print("action")
                        model_dict[new_key] = pretrained_dict[key]

        # 5. 加载norm层权重
        if 'model.norm.weight' in pretrained_dict:
            #print("norm")
            model_dict['identity_norm.weight'] = pretrained_dict['model.norm.weight']
            model_dict['identity_norm.bias'] = pretrained_dict['model.norm.bias']
            model_dict['action_norm.weight'] = pretrained_dict['model.norm.weight']
            model_dict['action_norm.bias'] = pretrained_dict['model.norm.bias']

        # 6. 加载原始head权重到identity_head
        if 'model.head.weight' in pretrained_dict:
            model_dict['identity_head.weight'] = pretrained_dict['model.head.weight']
            model_dict['identity_head.bias'] = pretrained_dict['model.head.bias']


        # 更新模型权重
        #print(model_dict.keys())
        self.load_state_dict(model_dict, strict=True)

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
        return x, B, T, W

    def forward(self, x):
        with no_grad():
            x, B, T, W = self.forward_features(x)
            # print("x, B, T, W: ", x.shape, B, T, W)
            # 通过共享的特征提取层
            for blk in self.shared_blocks:
                x = blk(x, B, T, W)


        # 身份分支
        with no_grad():
            identity_features = x.clone()
            for blk in self.identity_blocks:
                identity_features = blk(identity_features, B, T, W)
            identity_features = self.identity_norm(identity_features)
            identity_out = self.identity_head(identity_features[:, 0])

        # 动作分支
        action_features = x.clone()
        for blk in self.action_blocks:
            action_features = blk(action_features, B, T, W)
        action_features = self.action_norm(action_features)
        action_out = self.action_head(action_features[:, 0])

        return identity_out, identity_features[:, 0], action_out, action_features[:, 0]

@MODEL_REGISTRY.register()
class  TimeSformer_IA(nn.Module):
    def __init__(self, original_model, num_classes_identity=321, num_classes_action=4, split_layer=6):
        super(TimeSformer_IA, self).__init__()

        # 1. 加载共享组件
        self.patch_embed = original_model.model.patch_embed
        self.pos_drop = original_model.model.pos_drop
        self.time_drop = original_model.model.time_drop if hasattr(original_model.model, 'time_drop') else None
        self.cls_token = original_model.model.cls_token
        self.pos_embed = original_model.model.pos_embed
        self.attention_type = original_model.model.attention_type
        self.dropout = nn.Dropout(0.)
        if hasattr(original_model.model, 'time_embed'):
            self.time_embed = original_model.model.time_embed

        # 2. 共享层 (前split_layer层)
        self.shared_blocks = nn.ModuleList(list(original_model.model.blocks[:split_layer]))

        # 3. 身份分支 (split_layer层之后)
        self.identity_blocks = nn.ModuleList(list(original_model.model.blocks[split_layer:]))
        self.identity_norm = original_model.model.norm
        self.identity_head = nn.Linear(original_model.model.embed_dim, num_classes_identity)

        # 4. 动作分支 (split_layer层之后)
        self.action_blocks = nn.ModuleList(list(original_model.model.blocks[split_layer:]))
        self.action_norm = original_model.model.norm
        self.action_head = nn.Linear(original_model.model.embed_dim, num_classes_action)

        # 5. 初始化新添加的层
        self._init_weights_for_new_layers()

        # 6. 加载预训练权重
        self.load_pretrained_weights(original_model, split_layer)

        # 7. adapter
        self.adapter_id = TaskAdapter(768)
        self.adapter_act = TaskAdapter(768)

    def _init_weights_for_new_layers(self):
        # 初始化分类头
        nn.init.normal_(self.identity_head.weight, std=0.02)
        nn.init.constant_(self.identity_head.bias, 0)
        nn.init.normal_(self.action_head.weight, std=0.02)
        nn.init.constant_(self.action_head.bias, 0)

    def load_pretrained_weights(self, original_model, split_layer):
        # 获取原始模型的预训练权重
        pretrained_dict = original_model.state_dict()
        model_dict = self.state_dict()

        # 1. 加载共享组件权重
        shared_keys = ['patch_embed.proj.weight', 'patch_embed.proj.bias',
                       'pos_embed', 'cls_token']
        if hasattr(original_model.model, 'time_embed'):
            shared_keys.append('time_embed')

        for key in shared_keys:
            if f'model.{key}' in pretrained_dict and key in model_dict:
                #print("shared")
                #print(pretrained_dict[f'model.{key}'])
                model_dict[key] = pretrained_dict[f'model.{key}']

        # 2. 加载共享层权重
        for i in range(split_layer):
            for key in pretrained_dict:
                if f'model.blocks.{i}.' in key:
                    #print("shared weights")
                    new_key = key.replace(f'model.blocks.{i}.', f'shared_blocks.{i}.')
                    if new_key in model_dict:
                        # print("share")
                        model_dict[new_key] = pretrained_dict[key]
        # print("len: ", len(original_model.model.blocks)) # 12
        # 3. 加载身份分支权重
        for i in range(split_layer, len(original_model.model.blocks)):
            for key in pretrained_dict:
                if f'model.blocks.{i}.' in key:
                    #print("identity weights")
                    #print(f'identity_blocks.{i - split_layer}.')
                    new_key = key.replace(f'model.blocks.{i}.', f'identity_blocks.{i - split_layer}.')
                    if new_key in model_dict:
                        #print("identity")
                        model_dict[new_key] = pretrained_dict[key]

        # 4. 加载动作分支权重 (与身份分支相同初始化)
        for i in range(split_layer, len(original_model.model.blocks)):
            #print("i: ", i)
            for key in pretrained_dict:
                #print("key: ", key)
                if f'model.blocks.{i}.' in key:
                    #ori_key = f'model.blocks.{i}.'
                    #print(f'action_blocks.{i - split_layer}.')
                    new_key = key.replace(f'model.blocks.{i}.', f'action_blocks.{i - split_layer}.')
                    #print("new: ", new_key)
                    if new_key in model_dict:
                        #print("action")
                        model_dict[new_key] = pretrained_dict[key]

        # 5. 加载norm层权重
        if 'model.norm.weight' in pretrained_dict:
            #print("norm")
            model_dict['identity_norm.weight'] = pretrained_dict['model.norm.weight']
            model_dict['identity_norm.bias'] = pretrained_dict['model.norm.bias']
            model_dict['action_norm.weight'] = pretrained_dict['model.norm.weight']
            model_dict['action_norm.bias'] = pretrained_dict['model.norm.bias']

        # 6. 加载原始head权重到identity_head
        if 'model.head.weight' in pretrained_dict:
            model_dict['identity_head.weight'] = pretrained_dict['model.head.weight']
            model_dict['identity_head.bias'] = pretrained_dict['model.head.bias']

        # 更新模型权重
        #print(model_dict.keys())
        self.load_state_dict(model_dict, strict=True)

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0, 0, :].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0, 1:, :].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:, 1:]
            x = rearrange(x, '(b t) n m -> (b n) t m', b=B, t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m', b=B, t=T)
            x = torch.cat((cls_tokens, x), dim=1)
        return x, B, T, W

    def forward(self, x):
        with no_grad():
            x, B, T, W = self.forward_features(x)
            # print("x, B, T, W: ", x.shape, B, T, W)
            # 通过共享的特征提取层
            for blk in self.shared_blocks:
                x = blk(x, B, T, W)

        # adapter

        # 身份分支
        #with no_grad():
        identity_features = self.adapter_id(x.clone())
        for blk in self.identity_blocks:
            identity_features = blk(identity_features, B, T, W)
        identity_features = self.identity_norm(identity_features)
        identity_out = self.identity_head(identity_features[:, 0])

        # 动作分支
        action_features = self.adapter_act(x.clone())
        for blk in self.action_blocks:
            action_features = blk(action_features, B, T, W)
        action_features = self.action_norm(action_features)
        action_out = self.action_head(action_features[:, 0])

        # Full TLD Features
        return identity_out, identity_features[:, 1:], action_out, action_features[:, 1:]
        # Global Features
        # return identity_out, identity_features[:, 0], action_out, action_features[:, 0]



@MODEL_REGISTRY.register()
class TimeSformer_slot(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_action=4, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer_slot, self).__init__()
        self.pretrained=True
        self.model = VisionTransformer_slot(img_size=img_size, num_classes=num_classes, num_action=4, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)
        if self.pretrained:
            load_pretrained(self.model, num_classes=self.model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter, img_size=img_size, num_frames=num_frames, num_patches=self.num_patches, attention_type=self.attention_type, pretrained_model=pretrained_model)

        self.agg_block = AggregationBlock(num_latents=4, weight_tie_layers=False, depth=4)

        self.identity_class = num_classes
        self.action_class = num_action
        #if head_type == 'linear':
        self.head = nn.Linear(768, 321 + 4) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.head.weight.data.mul_(0.001)
        self.head.bias.data.mul_(0.001)
        self.fc_dropout = nn.Dropout(p=0.1) # if fc_drop_rate > 0 else nn.Identity()
        # else:
        #     #mlp
        #     self.head = MLPHead(embed_dim, num_classes + self.num_scene_classes, hidden_dim=512) if num_classes > 0 else nn.Identity()
        #
        #     trunc_normal_(self.head.fc1.weight, std=.02)
        #     trunc_normal_(self.head.fc2.weight, std=.02)
        #     self.apply(self._init_weights)
        #     self.head.fc2.weight.data.mul_(init_scale)
        #     self.head.fc2.bias.data.mul_(init_scale)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        with no_grad():
            x, _ = self.model(x)
        slots, attn = self.agg_block(x)
        #print("slots: ", slots.shape)  # slots:  torch.Size([4, 4, 768])
        #print("attn: ", attn.shape)    # attn:  torch.Size([16, 4, 3920])
        bs, num_slots, _ = slots.size()  # bs=4
        slots = slots.reshape(-1, 768)  # 将 (bs, num_slots, 768) 重塑为 (bs * num_slots, 768)。
        #print("slots: ", slots.shape)  # slots:  torch.Size([16, 768])
        # 这样做的目的是为了方便后续的全连接层（self.head）处理
        slots_head = self.head(self.fc_dropout(slots))
        #print("slots_head: ", slots_head.shape)  # slots_head:  torch.Size([16, 325])

        slot_probs = F.softmax(slots_head, dim=-1).view(bs, num_slots, -1)
        #print("slot_probs: ", slot_probs.shape)  # slot_probs:  torch.Size([4, 4, 325])

        # action_softmax_output = slot_probs[:, :, :self.identity_class]
        # scene_softmax_output = slot_probs[:, :, self.identity_class:self.identity_class + self.action_class]

        identity_softmax_output = slot_probs[:, :, :self.identity_class]
        action_softmax_output = slot_probs[:, :, self.identity_class:self.identity_class + self.action_class]

        identity_max_slot_indices = torch.argmax(identity_softmax_output.max(dim=-1).values, dim=1)
        action_max_slot_indices = torch.argmax(action_softmax_output.max(dim=-1).values, dim=1)

        identity_feat = slots.view(bs, num_slots, -1)[torch.arange(bs), identity_max_slot_indices]
        action_feat = slots.view(bs, num_slots, -1)[torch.arange(bs), action_max_slot_indices]
        identity_logit = slots_head.view(bs, num_slots, -1)[torch.arange(bs), identity_max_slot_indices]
        action_logit = slots_head.view(bs, num_slots, -1)[torch.arange(bs), action_max_slot_indices]

        # mask_predictions = self.mask_predictor(slots)
        #print("xxxxxxxx")

        return (identity_feat, action_feat), (identity_logit, action_logit, attn), (slots_head, slots)



class TrainLoss(nn.Module):
    """
    https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/models/matcher.py#L12
    """

    def __init__(self,
                 criterion: torch.nn.Module, num_identity_classes: int, num_action_classes: int,
                 ):
        super().__init__()
        self.criterion = criterion
        self.num_identity_classes = num_identity_classes
        self.num_action_classes = num_action_classes
        self.slot_matching_method = 'matching'

    # output
    # return (identity_feat, action_feat), (identity_logit, action_logit, attn), (slots_head, slots)

    def forward(self, model, backbone_output, identity_target, action_target):
        # slot_action_head : (bs x num_slots) x action_classes
        # slots_scene_head : (bs x num_slots) x scene_classes

        # (action_feat, scene_feat), (action_logit, scene_logit, attn), (slots_head, slots, mask_predictions)
        _, (action_output, identity_output, attn), (slots_head, slots) = backbone_output
        #print("action_output: ", action_output.shape)
        #print("identity_output: ", identity_output.shape)
        # print("attn: ", attn.shape)  # torch.Size([16, 4, 3920])
        #print("slots_head: ", slots_head.shape)  # slots_head:  torch.Size([16, 325])
        device = slots_head.device
        dtype = slots_head.dtype

        bs = identity_target.shape[0]
        num_latent = int(slots_head.shape[0] / bs)
        #print("num_latent", num_latent)  # 4
        num_head = attn.size(0) // bs
        #print("num_head", num_head)   # 4

        # attn mean per head
        attn = reduce(attn, '(bs num_head) num_latent dim -> bs num_latent dim', 'mean', num_head=num_head)
        # print("attn: ", attn.shape)  # torch.Size([4, 4, 3920])
        # mask_predictions = mask_predictions.reshape(bs, num_latent, -1)




        # var = -1e10 - float(1)
        # inf_tensor = torch.full((action_target.size(0), self.num_action_classes), var, device=device)


        action_target += self.num_identity_classes

        slots_head_sfmax = slots_head.softmax(-1)
        all_indices = []

        for i in range(bs):
            # Compute cost for each query for scene and action for the current image
            cost_identity_class = -slots_head_sfmax[i * num_latent:(i + 1) * num_latent, identity_target[i]]
            cost_action_class = -slots_head_sfmax[i * num_latent:(i + 1) * num_latent, action_target[i]]

            # Concatenate the two costs
            combined_cost = torch.cat([cost_identity_class.unsqueeze(-1), cost_action_class.unsqueeze(-1)], dim=1)

            # Use Hungarian algorithm on the combined cost
            indices = linear_sum_assignment(combined_cost.detach().cpu())
            all_indices.append(indices)

        # Convert the list of indices into the desired format
        all_indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in
                       all_indices]

        identity_loss = torch.tensor([0], device=device, dtype=dtype)
        action_loss = torch.tensor([0], device=device, dtype=dtype)
        selected_slots = []
        action_logit = []
        identity_logit = []
        slots_head = slots_head.view(bs, num_latent, -1)


        for batch_idx, (slot_indices, label_indices) in enumerate(all_indices):
            for s_idx, l_idx in zip(slot_indices, label_indices):
                if l_idx == 0:  # action
                    identity_loss += F.cross_entropy(slots_head[batch_idx, s_idx], identity_target[batch_idx])
                    identity_logit.append(slots_head[batch_idx, s_idx])
                    selected_slots.append((batch_idx, int(s_idx)))

                elif l_idx == 1:  # scene
                    #if self.scene_criterion == "CE":
                    action_loss += F.cross_entropy(slots_head[batch_idx, s_idx], action_target[batch_idx])
                    action_logit.append(slots_head[batch_idx, s_idx])
                    selected_slots.append((batch_idx, int(s_idx)))

        identity_loss /= bs
        action_loss /= bs


        slots = slots.reshape(bs, num_latent, -1)
        normed_slots = F.normalize(slots, p=2, dim=2)
        cosine_sim_matrix = torch.bmm(normed_slots, normed_slots.transpose(1, 2))
        identity = torch.eye(cosine_sim_matrix.size(1)).to(cosine_sim_matrix.device)
        cosine_sim_matrix = cosine_sim_matrix * (1 - identity)
        cosine_loss = (cosine_sim_matrix.sum(dim=(1, 2)) / (
                    cosine_sim_matrix.size(1) * (cosine_sim_matrix.size(1) - 1))).mean()

        total_loss = identity_loss + action_loss + cosine_loss
        return total_loss, \
            torch.stack(identity_logit), \
            torch.stack(action_logit), \
            {'identity_loss': identity_loss.item(),
             'action_loss': action_loss.item(),
             'cosine_loss': cosine_loss.item()}, action_output, identity_output

class TaskAdapter(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim),
        )
    def forward(self,x):
        return self.adapter(x) + x