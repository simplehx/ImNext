import torch.nn as nn
import torch
import torch.nn.functional as F

class IrrAttention(nn.Module):
    def __init__(self, n_head, d_model, poi_dim, time_dim, distance_dim, dropout):
        super().__init__()

        self.n_head = n_head
        self.poi_dim = poi_dim
        self.time_dim = time_dim
        self.distance_dim = distance_dim

        self.poi_w_q = nn.Linear(d_model, n_head * poi_dim, bias=False)
        self.poi_w_k = nn.Linear(d_model, n_head * poi_dim, bias=False)
        self.poi_w_v = nn.Linear(d_model, n_head * poi_dim, bias=False)

        self.time_w_q = nn.Linear(d_model, n_head * time_dim, bias=False)
        self.time_w_k = nn.Linear(d_model, n_head * time_dim, bias=False)
        self.time_w_v = nn.Linear(d_model, n_head * time_dim, bias=False)
        
        self.distance_w_q = nn.Linear(d_model, n_head * distance_dim, bias=False)
        self.distance_w_k = nn.Linear(d_model, n_head * distance_dim, bias=False)
        self.distance_w_v = nn.Linear(d_model, n_head * distance_dim, bias=False)

        self.q_fc = nn.Sequential(
            nn.Linear(n_head * poi_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )

        self.iq_fc = nn.Sequential(
            nn.Linear(n_head * time_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )

        self.dq_fc = nn.Sequential(
            nn.Linear(n_head * distance_dim, d_model, bias=False),
            nn.Dropout(dropout)
        )

        self.attention = ScaledDotProductAttention(temperature=poi_dim ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

        self.q_output_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.iq_output_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dq_output_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, poi_inputs, time_interval, distance_interval, mask=None):
        poi_dim, time_dim, distance_dim, n_head = self.poi_dim, self.time_dim, self.distance_dim, self.n_head
        batch_size, len_poi, len_time_interval, len_distance_interval = poi_inputs.size(0), poi_inputs.size(1), time_interval.size(1), distance_interval.size(1)

        poi_residual = poi_inputs
        time_residual = time_interval
        distance_residual = distance_interval

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        
        poi_q = self.poi_w_q(poi_inputs).view(batch_size, len_poi, n_head, poi_dim)
        poi_k = self.poi_w_k(poi_inputs).view(batch_size, len_poi, n_head, poi_dim)
        poi_v = self.poi_w_v(poi_inputs).view(batch_size, len_poi, n_head, poi_dim)

        poi_merge = poi_inputs[:, :-1, :] + poi_inputs[:, 1:, :]
        
        time_q = self.time_w_q(poi_merge + time_interval).view(batch_size, len_time_interval, n_head, time_dim)
        time_k = self.time_w_k(poi_merge + time_interval).view(batch_size, len_time_interval, n_head, time_dim)
        time_v = self.time_w_v(time_interval).view(batch_size, len_time_interval, n_head, time_dim)

        distance_q = self.distance_w_q(poi_merge + distance_interval).view(batch_size, len_distance_interval, n_head, distance_dim)
        distance_k = self.distance_w_k(poi_merge + distance_interval).view(batch_size, len_distance_interval, n_head, distance_dim)
        distance_v = self.distance_w_v(distance_interval).view(batch_size, len_distance_interval, n_head, distance_dim)
        
        # Transpose for attention dot product: b x n x lq x dv
        poi_q, poi_k, poi_v = poi_q.transpose(1, 2), poi_k.transpose(1, 2), poi_v.transpose(1, 2)
        time_q, time_k, time_v = time_q.transpose(1, 2), time_k.transpose(1, 2), time_v.transpose(1, 2)
        distance_q, distance_k, distance_v = distance_q.transpose(1, 2), distance_k.transpose(1, 2), distance_v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        poi_output, time_output, distance_output, attn1, attn2, attn3 = self.attention(poi_q, poi_k, poi_v, time_q, time_k, time_v, distance_q, distance_k, distance_v, mask=mask)
        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        poi_output = poi_output.transpose(1, 2).contiguous().view(batch_size, len_poi, -1)
        time_output = time_output.transpose(1, 2).contiguous().view(batch_size, len_time_interval, -1)
        distance_output = distance_output.transpose(1, 2).contiguous().view(batch_size, len_distance_interval, -1)

        poi_output = self.q_fc(poi_output)
        time_output = self.iq_fc(time_output)
        distance_output = self.dq_fc(distance_output)

        poi_output += poi_residual
        time_output += time_residual
        distance_output += distance_residual

        poi_output = self.q_output_norm(poi_output)
        time_output = self.iq_output_norm(time_output)
        distance_output = self.dq_output_norm(distance_output)

        return poi_output, time_output, distance_output, attn1, attn2, attn3
    

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        self.temperature = temperature
        self.attn1_dropout = nn.Dropout(attn_dropout)
        self.attn2_dropout = nn.Dropout(attn_dropout)
        self.attn3_dropout = nn.Dropout(attn_dropout)

    def forward(self, poi_q, poi_k, poi_v, time_q, time_k, time_v, distance_q, distance_k, distance_v, mask=None):

        attn1 = torch.matmul(poi_q / self.temperature, poi_k.transpose(2, 3))
        attn2 = torch.matmul(time_q / self.temperature, time_k.transpose(2, 3))
        attn3 = torch.matmul(distance_q / self.temperature, distance_k.transpose(2, 3))
        
        if mask is not None:
            attn1 = attn1.masked_fill(mask == 0, -1e9)
        attn1 = self.attn1_dropout(F.softmax(attn1, dim=-1))
        attn2 = self.attn2_dropout(F.softmax(attn2, dim=-1))
        attn3 = self.attn3_dropout(F.softmax(attn3, dim=-1))

        poi_output = torch.matmul(attn1, poi_v)
        time_output = torch.matmul(attn2, time_v)
        distance_output = torch.matmul(attn3, distance_v)

        poi_output[...,1:,:] += torch.matmul(attn2, poi_v[...,:-1,:])
        poi_output[...,1:,:] += torch.matmul(attn3, poi_v[...,:-1,:])
        return poi_output, time_output, distance_output, attn1, attn2, attn3