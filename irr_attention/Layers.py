"""
this file modify from github： https://github.com/jadore801120/attention-is-all-you-need-pytorch
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
# from IrrAttention import IrrAttention
# from IrrAttention import IrrAttention
from irr_attention.IrrAttention import IrrAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, poi_dim, time_dim, distance_dim, dropout):
        super(EncoderLayer, self).__init__()
        self.slf_attn = IrrAttention(n_head, d_model, poi_dim, time_dim, distance_dim, dropout=dropout)
        self.poi_pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.time_pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.distance_pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, poi_input, time_input, distance_input, slf_attn_mask=None):
        poi_output, time_output, distance_output, enc_attn, time_attn, distance_attn = self.slf_attn(poi_input, time_input, distance_input, mask=slf_attn_mask)
        poi_output = self.poi_pos_ffn(poi_output)
        time_output = self.time_pos_ffn(time_output)
        distance_output = self.distance_pos_ffn(distance_output)
        return poi_output, time_output, distance_output, enc_attn, time_attn, distance_attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x
