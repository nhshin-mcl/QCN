import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast

from timm.models.layers import DropPath
from timm.models.layers import trunc_normal_tf_ as trunc_normal_

import math


def get_activation(name):
    if name is None or name.lower() == 'none':
        return nn.Identity()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()


def get_norm(name, channels):
    if name is None or name.lower() == 'none':
        return nn.Identity()

    if name.lower() == 'syncbn':
        return nn.SyncBatchNorm(channels, eps=1e-3, momentum=0.01)

    if name.lower() == 'bn1d':
        return nn.BatchNorm1d(channels)

    if name.lower() == 'bn2d':
        return nn.BatchNorm2d(channels)

    if name.lower() == 'bn3d':
        return nn.BatchNorm3d(channels)

    if name.lower() == 'ln':
        return nn.GroupNorm(1, channels)

    if name.lower() == 'gn':
        return nn.GroupNorm(5, channels)


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, norm=None, act=None,
                 conv_type='2d', conv_init='he_normal', norm_init=1.0):
        super().__init__()

        if conv_type == '2d':
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        elif conv_type == '1d':
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.act = get_activation(act)

        if conv_init == 'normal':
            nn.init.normal_(self.conv.weight, std=.02)
        elif conv_init == 'trunc_normal':
            trunc_normal_(self.conv.weight, std=.02)
        elif conv_init == 'he_normal':
            trunc_normal_(self.conv.weight, std=math.sqrt(2.0 / in_channels))
        elif conv_init == 'xavier_uniform':
            nn.init.xavier_uniform_(self.conv.weight)
        if bias:
            nn.init.zeros_(self.conv.bias)

        if norm != None and norm.lower() != 'none':
            nn.init.constant_(self.norm.weight, norm_init)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class AttentionOperation(nn.Module):
    def __init__(self, channels_v, num_heads):
        super().__init__()

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('bhdl,bhdm->bhlm', query, key)
        _, _, _, L_sim = similarity_logits.size()
        similarity_logits = nn.LayerNorm([num_heads, L, L_sim]).cuda()(similarity_logits)

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
        retrieved_value = torch.einsum('bhlm,bhdm->bhdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = nn.LayerNorm([num_heads * C, L]).cuda()(retrieved_value)

        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value

class MaskedAttentionOperation(nn.Module):
    def __init__(self, mask_value, num_heads):
        super().__init__()
        self.mask_value = mask_value
        self.num_heads = num_heads

    def forward(self, query, key, value):
        N, _, _, L = query.shape
        _, num_heads, C, _ = value.shape
        similarity_logits = torch.einsum('BRCL,BRCM->BRLM', query, key)
        similarity_logits = nn.LayerNorm([L, L]).cuda()(similarity_logits)

        for i in range(self.num_heads):

            # delete diagonal
            mask_idx = torch.stack([torch.arange(N).repeat_interleave(L),
                                    torch.arange(L).repeat(N),
                                    torch.arange(L).repeat(N)]).cuda()
            similarity_logits[mask_idx[0], i, mask_idx[1], mask_idx[2]] = self.mask_value

        with autocast(enabled=False):
            attention_weights = F.softmax(similarity_logits.float(), dim=-1)
            attention_weights = torch.where(torch.isnan(attention_weights) == True, 0, attention_weights)

        retrieved_value = torch.einsum(
            'BRlm,BRdm->BRdl', attention_weights, value)
        retrieved_value = retrieved_value.reshape(N, num_heads * C, L)
        retrieved_value = nn.LayerNorm([num_heads * C, L]).cuda()(retrieved_value)

        retrieved_value = F.gelu(retrieved_value)
        return retrieved_value

class CrossAlignmentLayer(nn.Module):
    def __init__(
            self,
            in_channel_pixel=2048,
            in_channel_query=256,
            base_filters=128,
            num_heads=8,
            bottleneck_expansion=2,
            key_expansion=1,
            value_expansion=2,
            drop_path_prob=0.1,
    ):
        super().__init__()

        self._num_heads = num_heads
        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        self.drop_path_kmeans = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5
        self._to_query_space = ConvBN(in_channel_query, self._bottleneck_channels, kernel_size=1, bias=False,
                                      norm='ln', act='gelu', conv_type='1d')

        self._to_input_space = ConvBN(in_channel_pixel, self._bottleneck_channels, kernel_size=1, bias=False,
                                      norm='ln', act='gelu', conv_type='1d')

        # query self attention
        self._query_to_qkv = ConvBN(self._bottleneck_channels, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False,
                                    norm='ln', act=None, conv_type='1d')
        trunc_normal_(self._query_to_qkv.conv.weight, std=initialization_std)

        self._query_self_attention = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                      norm='ln', act=None, conv_type='1d', norm_init=0.0)

        # FFN

        self._query_ffn_1 = ConvBN(in_channel_query, in_channel_query * 2, kernel_size=1, bias=False,
                                   norm='ln', act='gelu', conv_type='1d')
        self._query_ffn_2 = ConvBN(in_channel_query * 2, in_channel_query, kernel_size=1, bias=False,
                                   norm='ln', act=None, conv_type='1d', norm_init=0.0)

        # Cross
        self._query_q_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth, kernel_size=1, bias=False, norm='ln', act=None, conv_type='1d')
        trunc_normal_(self._query_q_conv_bn.conv.weight, std=initialization_std)

        self._pixel_kv_conv_bn = ConvBN(self._bottleneck_channels, self._total_key_depth + self._total_value_depth, kernel_size=1, bias=False, norm='ln', act=None, conv_type='1d')
        trunc_normal_(self._pixel_kv_conv_bn.conv.weight, std=initialization_std)
        self._cross_attn = AttentionOperation(channels_v=self._total_value_depth, num_heads=num_heads)

        self._query_conv3_bn_cross = ConvBN(self._total_value_depth, in_channel_query, kernel_size=1, bias=False,
                                            norm='ln', act=None, conv_type='1d', norm_init=0.0)

    def forward(self, input_f, query_f, self_attention=True):
        N, C, L = query_f.shape
        _, Ci, Li = input_f.shape
        input_space = self._to_input_space(input_f)
        query_space = self._to_query_space(query_f)

        # Cross update
        query_q = self._query_q_conv_bn(query_space)
        input_kv = self._pixel_kv_conv_bn(input_space)
        input_k, input_v = torch.split(input_kv, [self._total_key_depth, self._total_value_depth], dim=1)

        query_q = query_q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        input_k = input_k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, Li)
        input_v = input_v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, Li)

        cross_attn_update = self._cross_attn(query_q, input_k, input_v)
        cross_attn_update = self._query_conv3_bn_cross(cross_attn_update)
        query_f = query_f + self.drop_path_kmeans(cross_attn_update)

        if self_attention:
            # self update
            query_qkv = self._query_to_qkv(F.gelu(query_f))
            query_q, query_k, query_v = torch.split(query_qkv, [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
            query_q = query_q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
            query_k = query_k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
            query_v = query_v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)
            self_attn_update = self._query_self_attention(query_q, query_k, query_v)
            self_attn_update = self._query_conv3_bn(self_attn_update)
            query_f = query_f + self.drop_path_attn(self_attn_update)
            query_f = F.gelu(query_f)

            # FFN.
            ffn_update = self._query_ffn_1(query_f)
            ffn_update = self._query_ffn_2(ffn_update)
            query_f = query_f + self.drop_path_ffn(ffn_update)
            query_f = F.gelu(query_f)

        return query_f


class FeatureAlignmentLayer(nn.Module):
    def __init__(
            self,
            in_channel_feature=256,
            base_filters=128,
            bottleneck_expansion=2,
            key_expansion=1,
            num_heads=8,
            value_expansion=2,
            drop_path_prob=0.0,
    ):
        super().__init__()

        self._bottleneck_channels = int(round(base_filters * bottleneck_expansion))
        self._total_key_depth = int(round(base_filters * key_expansion))
        self._total_value_depth = int(round(base_filters * value_expansion))

        self.drop_path_attn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        self.drop_path_ffn = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()

        initialization_std = self._bottleneck_channels ** -0.5

        # Masked self attention
        self._to_z_space = ConvBN(in_channel_feature, in_channel_feature, kernel_size=1, bias=False, norm='ln', act='gelu', conv_type='1d')
        self._to_qkv = ConvBN(in_channel_feature, self._total_key_depth * 2 + self._total_value_depth, kernel_size=1, bias=False, norm='ln', act=None, conv_type='1d')
        trunc_normal_(self._to_qkv.conv.weight, std=initialization_std)

        self._num_heads = num_heads
        self._masked_self_attention = MaskedAttentionOperation(mask_value=-torch.inf, num_heads = self._num_heads)
        self._to_out = ConvBN(self._total_value_depth, in_channel_feature, kernel_size=1, bias=False,
                              norm='ln', act=None, conv_type='1d', norm_init=0.0)

        # Masked self attention FFN
        self._self_masked_attn_ffn_1 = ConvBN(in_channel_feature, in_channel_feature * 2, kernel_size=1, bias=False,
                                              norm='ln', act='gelu', conv_type='1d')
        self._self_masked_attn_ffn_2 = ConvBN(in_channel_feature * 2, in_channel_feature, kernel_size=1, bias=False,
                                              norm='ln', act=None, conv_type='1d', norm_init=0.0)


    def forward(self, f):
        N, C, L = f.shape
        z_space = self._to_z_space(F.gelu(f))

        # masked self attention
        z_qkv = self._to_qkv(z_space)
        q, k, v = torch.split(z_qkv, [self._total_key_depth, self._total_key_depth, self._total_value_depth], dim=1)
        q = q.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        k = k.reshape(N, self._num_heads, self._total_key_depth // self._num_heads, L)
        v = v.reshape(N, self._num_heads, self._total_value_depth // self._num_heads, L)

        self_masked_attn_update = self._masked_self_attention(q, k, v)
        self_masked_attn_update = self._to_out(self_masked_attn_update)

        f = f + self.drop_path_attn(self_masked_attn_update)
        f = F.gelu(f)

        # FFN.
        ffn_update = self._self_masked_attn_ffn_1(f)
        ffn_update = self._self_masked_attn_ffn_2(ffn_update)
        f = f + self.drop_path_ffn(ffn_update)
        f = F.gelu(f)

        return f


class ComparisonTransformerLayer(nn.Module):

    def __init__(
            self,
            num_heads=8,
            dec_layers=[2, 2, 2],
            query_dim=256,
            key_value_dim=256,
            num_queries=101,
            drop_path_prob=0.5,
    ):

        super().__init__()

        # The number of CT modules
        self._num_blocks = dec_layers

        # Feature self update module
        self._init_feature_alignment_layer = FeatureAlignmentLayer(in_channel_feature=key_value_dim,
                                                                   base_filters=key_value_dim // 2,
                                                                   bottleneck_expansion=2,
                                                                   key_expansion=1,
                                                                   value_expansion=2,
                                                                   drop_path_prob=drop_path_prob, )

        self._feature_alignment_layers = nn.ModuleList()
        for index, _ in enumerate(dec_layers):
            for _ in range(self._num_blocks[index]):
                self._feature_alignment_layers.append(
                    FeatureAlignmentLayer(in_channel_feature=key_value_dim,
                                          base_filters=key_value_dim // 2,
                                          bottleneck_expansion=2,
                                          key_expansion=1,
                                          value_expansion=2,
                                          drop_path_prob=drop_path_prob, )
                )

        # feature pivot cross update & pivot self update modules
        self._rpt_cross_alignment_layers = nn.ModuleList()
        for index, _ in enumerate(dec_layers):
            for _ in range(self._num_blocks[index]):
                self._rpt_cross_alignment_layers.append(
                    CrossAlignmentLayer(in_channel_pixel=key_value_dim,
                                        in_channel_query=query_dim,
                                        base_filters=key_value_dim // 2,
                                        num_heads=num_heads,
                                        bottleneck_expansion=2,
                                        key_expansion=1,
                                        value_expansion=2,
                                        drop_path_prob=drop_path_prob)
                )

        # pivot feature cross update module
        self._f_cross_alignment_layers = nn.ModuleList()
        for index, _ in enumerate(dec_layers):
            for _ in range(self._num_blocks[index]):
                self._f_cross_alignment_layers.append(
                    CrossAlignmentLayer(in_channel_pixel=query_dim,
                                        in_channel_query=key_value_dim,
                                        base_filters=key_value_dim // 2,
                                        num_heads=num_heads,
                                        bottleneck_expansion=2,
                                        key_expansion=1,
                                        value_expansion=2,
                                        drop_path_prob=drop_path_prob)
                )

        # initialize score pivots
        self._num_queries = num_queries
        self._cluster_centers = nn.Embedding(query_dim, num_queries)
        trunc_normal_(self._cluster_centers.weight, std=0.5)

    def forward(self, x):
        B = x.shape[0]
        cluster_centers = self._cluster_centers.weight.unsqueeze(0).repeat(B, 1, 1)  # N x C x L /  N X 256 X 101

        current_transformer_idx = 0
        current_inverse_transformer_idx = 0
        current_feature_alignment_idx = 0

        # Initial feature self update
        x = self._init_feature_alignment_layer(f=x)

        for i in range(len(self._num_blocks)):

            # Feature pivot cross update
            # Pivot self update is done when self attention is True
            for _ in range(self._num_blocks[i]):
                cluster_centers = self._rpt_cross_alignment_layers[current_transformer_idx](input_f=x, query_f=cluster_centers, self_attention=True)
                current_transformer_idx += 1

            # Pivot feature cross update
            for _ in range(self._num_blocks[i]):
                x = self._f_cross_alignment_layers[current_inverse_transformer_idx](input_f=cluster_centers, query_f=x, self_attention=False)
                current_inverse_transformer_idx += 1

            # Feature self update
            if i < (len(self._num_blocks) - 1):
                for _ in range(self._num_blocks[i]):
                    x = self._feature_alignment_layers[current_feature_alignment_idx](f=x)
                    current_feature_alignment_idx += 1

        return x, cluster_centers
