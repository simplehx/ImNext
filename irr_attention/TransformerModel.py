"""
this file modify from github： https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import torch
import torch.nn as nn
import numpy as np
from irr_attention.Layers import EncoderLayer

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, n_layers, n_head, poi_dim, time_dim, distance_dim, d_model, d_inner, dropout, n_position=200, scale_emb=False):

        super().__init__()
        self.poi_position_enc = nn.Sequential(
            PositionalEncoding(poi_dim, n_position=n_position),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model, eps=1e-6)
        )
        self.time_position_enc = nn.Sequential(
            PositionalEncoding(time_dim, n_position=n_position),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model, eps=1e-6)
        )
        self.distance_position_enc = nn.Sequential(
            PositionalEncoding(distance_dim, n_position=n_position),
            nn.Dropout(p=dropout),
            nn.LayerNorm(d_model, eps=1e-6)
        )
        self.layer_stack = nn.ModuleList([EncoderLayer(d_model, d_inner, n_head, poi_dim, time_dim, distance_dim, dropout=dropout) for _ in range(n_layers)])
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, poi_input, src_mask, time_input, distance_input, return_attns=False):
        # -- Forward
        if self.scale_emb:
            poi_input *= self.d_model ** 0.5
        poi_input = self.poi_position_enc(poi_input)
        time_input = self.time_position_enc(time_input)
        distance_input = self.distance_position_enc(distance_input)
        for enc_layer in self.layer_stack:
            poi_input, time_input, distance_input, enc_attn, time_attn, distance_attn = enc_layer(poi_input, time_input, distance_input, slf_attn_mask=src_mask)


        return poi_input, time_input, distance_input
