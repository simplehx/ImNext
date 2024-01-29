import torch
import torch.nn as nn
from EAGAT import EAGAT
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from irr_attention.TransformerModel import Encoder
from torch_geometric.nn import norm

class TemporalEmbedding(nn.Module):
    def __init__(self, embedding_len, year_list):
        super().__init__()
        self.year2index = {year: index for index, year in enumerate(year_list)}
        self.year_emb = nn.Embedding(len(self.year2index), embedding_len)
        self.month_emb = nn.Embedding(12, embedding_len)
        self.day_emb = nn.Embedding(31, embedding_len)
        self.weekday_emb = nn.Embedding(7, embedding_len)
        self.hour_emb = nn.Embedding(24, embedding_len)
        self.min_emb = nn.Embedding(6, embedding_len)
    
    def forward(self, x):
        year, month, day, weekday, hour, min, sec = torch.chunk(x, dim=2, chunks=7)
        year, month, day, weekday, hour, min, sec = year.squeeze(2), month.squeeze(2), day.squeeze(2), weekday.squeeze(2), hour.squeeze(2), min.squeeze(2), sec.squeeze(2)
        for from_year, to_index in self.year2index.items():
            year = torch.where(year == from_year, to_index, year)
        
        year_embedding = self.year_emb(year)
        month_embedding = self.month_emb(month - 1)
        day_embedding = self.day_emb(day - 1)
        weekday_embedding = self.weekday_emb(weekday)
        hour_embedding = self.hour_emb(hour)
        min_embedding = self.min_emb(min // 10)
        # return month_embedding + day_embedding + hour_embedding + min_embedding
        return year_embedding + month_embedding + day_embedding + weekday_embedding + hour_embedding + min_embedding

class GraphLayer(nn.Module):
    def __init__(self, dimension, edge_dim, dropout):
        super().__init__()
        self.conv1 = EAGAT(dimension, dimension, edge_dim=edge_dim, dropout=dropout)
        self.conv1_norm = norm.LayerNorm(dimension)
        self.conv2 = EAGAT(dimension, dimension, edge_dim=edge_dim, dropout=dropout)
        self.conv2_norm = norm.LayerNorm(dimension)
        self.linear = nn.Sequential(
            nn.Linear(dimension, dimension),
            nn.LayerNorm(dimension),
            nn.ReLU()
        )
    
    def forward(self, x, edge_index, batch, edge_attr):
        x1 = self.conv1(x, edge_index, edge_attr)
        x2 = self.conv1_norm(x1)
        x3 = self.conv2(x2, edge_index, edge_attr)
        x4 = self.conv2_norm(x3)
        x5 = global_mean_pool(x4, batch)
        output = self.linear(x5)
        return output


class POIPredictionModel(nn.Module):
    def __init__(self, num_classes, poi_count, user_count, year_list, dropout, seq_len, layer, dimension):
        super().__init__()
        # poi embedding
        self.poi_embedding = nn.Embedding(num_embeddings=poi_count + 1, embedding_dim=dimension, padding_idx=poi_count)
        # user embedding
        self.user_embedding = nn.Embedding(num_embeddings=user_count, embedding_dim=dimension)
        # temporal embedding
        self.temporal_embedding = TemporalEmbedding(dimension, year_list)
        # time interval embedding
        self.time_interval_embedding = nn.Sequential(
            nn.Linear(1, dimension),
            nn.LayerNorm(dimension)
        )
        # distance embedding
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, dimension),
            nn.LayerNorm(dimension)
        )
        # trans model
        self.transformer_encoder = Encoder(n_layers=layer, n_head=3, poi_dim=dimension, time_dim=dimension, distance_dim=dimension, d_model=dimension, d_inner=256, dropout=dropout)
        self.trans_traj_pooling = nn.Sequential(
            nn.Linear(dimension, 1)
        )
        self.trans_distance_pooling = nn.Sequential(
            nn.Linear(dimension, 1)
        )
        self.trans_time_pooling = nn.Sequential(
            nn.Linear(dimension, 1)
        )
        # TrajectroyGraphModel
        self.tG_layer = GraphLayer(dimension, edge_dim=1, dropout=dropout)
        
        # DistanceGraphModel
        self.dG_layer = GraphLayer(dimension, edge_dim=1, dropout=dropout)
        
        # VisitedGraphModel
        self.vG_layer = GraphLayer(dimension, edge_dim=1, dropout=dropout)
        
        # LastVisitNeighborsGraphModel
        self.lG_layer = GraphLayer(dimension, edge_dim=1, dropout=dropout)
        
        # FriendsVisitGraphModel
        self.fG_layer = GraphLayer(dimension, edge_dim=1, dropout=dropout)
        
        self.shared_layer = nn.Sequential(
            nn.Linear(dimension, dimension * 2),
            nn.LayerNorm(dimension * 2),
            nn.ReLU(),
            nn.Linear(dimension * 2, dimension),
            nn.LayerNorm(dimension),
            nn.ReLU()
        )

        self.is_visited_pred_linear = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.LayerNorm(dimension // 2),
            nn.ReLU(),
            nn.Linear(dimension // 2, 2)
        )
        self.distance_pred_linear = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.LayerNorm(dimension // 2),
            nn.ReLU(),
            nn.Linear(dimension // 2, 1)
        )
        self.time_interval_pred_linear = nn.Sequential(
            nn.Linear(dimension, dimension // 2),
            nn.LayerNorm(dimension // 2),
            nn.ReLU(),
            nn.Linear(dimension // 2, 1)
        )
        self.next_poi_pred_linear = nn.Sequential(
            nn.Linear(dimension, dimension * 4),
            nn.LayerNorm(dimension * 4),
            nn.ReLU(),
            nn.Linear(dimension * 4, num_classes)
        )

    def forward(self, user_index, trans_input, time_input, time_interval_input, distance_input, trajectory_graph, distance_graph, last_visit_neighbors_graph, visited_graph, friends_visit_graph):
        # trans model
        poi_embedding = self.poi_embedding(trans_input)
        user_embedding = self.user_embedding(user_index)
        temporal_embedding = self.temporal_embedding(time_input)
        trans_input = poi_embedding + temporal_embedding
        distance_embedding = self.distance_embedding(distance_input.unsqueeze(-1))
        time_interval_embedding = self.time_interval_embedding(time_interval_input.unsqueeze(-1))

        poi_output, time_output, distance_output = self.transformer_encoder(trans_input, None, time_interval_embedding, distance_embedding)
        traj_weight = F.softmax(self.trans_traj_pooling(poi_output), dim=1)
        trans_output = torch.sum(traj_weight * poi_output, dim=1)
        distance_weight = F.softmax(self.trans_distance_pooling(distance_output), dim=1)
        distance_output = torch.sum(distance_weight * distance_output, dim=1)
        time_weight = F.softmax(self.trans_time_pooling(time_output), dim=1)
        time_output = torch.sum(time_weight * time_output, dim=1)

        # LastVisitNeighborsGraphModel
        x, edge_index, batch, edge_attr = last_visit_neighbors_graph.x, last_visit_neighbors_graph.edge_index, last_visit_neighbors_graph.batch, last_visit_neighbors_graph.edge_attr
        lG_output = self.lG_layer(self.poi_embedding(x), edge_index, batch, edge_attr)
        # FriendsVisitGraphModel
        x, edge_index, batch, edge_attr = friends_visit_graph.x, friends_visit_graph.edge_index, friends_visit_graph.batch, friends_visit_graph.edge_attr
        fG_output = self.fG_layer(self.poi_embedding(x), edge_index, batch, edge_attr)
        # TrajectroyGraphModel
        x, edge_index, batch, edge_attr = trajectory_graph.x, trajectory_graph.edge_index, trajectory_graph.batch, trajectory_graph.edge_attr
        tG_output = self.tG_layer(self.poi_embedding(x), edge_index, batch, edge_attr)
        # DistanceGraphModel
        x, edge_index, batch, edge_attr = distance_graph.x, distance_graph.edge_index, distance_graph.batch, distance_graph.edge_attr
        dG_output = self.dG_layer(self.poi_embedding(x), edge_index, batch, edge_attr)
        # VisitedGraphModel
        x, edge_index, batch, edge_attr = visited_graph.x, visited_graph.edge_index, visited_graph.batch, visited_graph.edge_attr
        vG_output = self.vG_layer(self.poi_embedding(x), edge_index, batch, edge_attr)
        
        shared_layer_input = trans_output + lG_output + fG_output + vG_output + dG_output + tG_output + user_embedding
        shared_layer_output = self.shared_layer(shared_layer_input)
        next_poi_pred_output = self.next_poi_pred_linear(shared_layer_output)

        is_visited_pred_output = self.is_visited_pred_linear(shared_layer_output + vG_output)
        distance_pred_output = self.distance_pred_linear(shared_layer_output + distance_output + dG_output)
        time_interval_pred_output = self.time_interval_pred_linear(shared_layer_output + time_output + tG_output)

        return is_visited_pred_output, distance_pred_output, time_interval_pred_output, next_poi_pred_output


if __name__ == '__main__':
    edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
    edge_attr = torch.tensor([[10], [10], [20], [20]], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index.t().contiguous())

    gat = EAGAT(1, 2, edge_dim=1)
    gat(x, edge_index.t().contiguous(), edge_attr)