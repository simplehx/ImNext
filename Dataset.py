import numpy as np
import torch
from torch_geometric.data.dataset import Dataset
from torch_geometric.data import Data as GraphData
from multiprocessing import Manager
import pickle
from torch_geometric.utils import to_undirected

class POIDataset(Dataset):
    def __init__(self, dataset_index_list, last_visit_graph_dict, txn):
        super(POIDataset, self).__init__()
        self.dataset_index_list = dataset_index_list
        self.dataset_len = len(dataset_index_list)
        self.last_visit_graph_dict = last_visit_graph_dict
        self.txn = txn

    def graph_trans(self, graph):
        x, edge_index, edge_attr = graph
        graph = GraphData(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr))
        return graph
    
    def get(self, index):
        dataset_index = self.dataset_index_list[index]
        user_index, poi_list, dt_split_list, time_interval_list, distance_interval_list, trajectory_time_graph, trajectory_distance_graph, visited_graph, friend_visited_graph, last_visit_poi, poi_label, distance_label, time_interval_label, is_visited_label = pickle.loads(self.txn.get(dataset_index.encode()))
        
        # graph
        trajectory_time_graph = self.graph_trans(trajectory_time_graph)
        trajectory_distance_graph = self.graph_trans(trajectory_distance_graph)
        visited_graph = self.graph_trans(visited_graph)
        friend_visited_graph = self.graph_trans(friend_visited_graph)
        last_visit_graph = self.graph_trans(self.last_visit_graph_dict[last_visit_poi])
        
        # user
        user_index = torch.tensor(user_index, dtype=torch.long)
        # label
        poi_label = torch.tensor(poi_label, dtype=torch.long)
        distance_label = torch.tensor(distance_label, dtype=torch.float)
        time_interval_label = torch.tensor(time_interval_label, dtype=torch.float)
        is_visited_label = torch.tensor(is_visited_label, dtype=torch.long)
        # seq
        poi_list = torch.from_numpy(poi_list)
        dt_split_list = torch.from_numpy(dt_split_list)
        time_interval_list = torch.from_numpy(time_interval_list)
        distance_interval_list = torch.from_numpy(distance_interval_list)
        data = {"user_index": user_index, "poi_list": poi_list, "dt_split_list": dt_split_list, "time_interval_list": time_interval_list, "distance_interval_list": distance_interval_list, 
                "trajectory_time_graph": trajectory_time_graph, "trajectory_distance_graph": trajectory_distance_graph, 
                "visited_graph": visited_graph, "friend_visited_graph": friend_visited_graph, "last_visit_graph": last_visit_graph, 
                "poi_label": poi_label, "distance_label": distance_label, "time_interval_label": time_interval_label, "is_visited_label": is_visited_label}

        return data
    

    def len(self):
        return self.dataset_len
    