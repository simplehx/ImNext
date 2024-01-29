import numpy as np
from tqdm import tqdm
import lmdb
import pickle

time_interval_scale, distance_scale = 72, 50
# time_interval_scale, distance_scale = 120, 130
use_scale = True
dataset = "gowalla"

def dataset_trans(index_list, txn, new_txn):
    for index in tqdm(index_list):
        user_index, poi_list, dt_split_list, time_interval_list, distance_interval_list, trajectory_time_graph, trajectory_distance_graph, visited_graph, friend_visited_graph, last_visit_poi, poi_label, distance_label, time_interval_label, is_visited_label = pickle.loads(txn.get(index.encode()))
        
        dt_split_list = np.array(dt_split_list, dtype=np.int64)
        poi_list = np.array(poi_list, dtype=np.int64)
        time_interval_list = np.array(time_interval_list, dtype=np.float32)
        distance_interval_list = np.array(distance_interval_list, dtype=np.float32)
        x, edge_index, edge_attr = visited_graph
        visited_graph = [np.array(x, dtype=np.int64), np.array(edge_index, dtype=np.int64), np.array(edge_attr, dtype=np.float32)]

        # graph
        # time_interval_label = time_interval_label / 3600
        # time_interval_list = np.array(time_interval_list, dtype=np.float32) / 3600

        x, edge_index, edge_attr = trajectory_time_graph
        edge_attr = np.array(edge_attr) / 3600
        edge_attr = np.where(edge_attr < time_interval_scale, edge_attr, time_interval_scale)
        trajectory_time_graph = [np.array(x, dtype=np.int64), np.array(edge_index, dtype=np.int64), np.array(edge_attr, dtype=np.float32)]

        x, edge_index, edge_attr = friend_visited_graph
        friend_visited_graph = [np.array(x, dtype=np.int64), np.array(edge_index, dtype=np.int64), np.array(edge_attr, dtype=np.float32)]
        
        x, edge_index, edge_attr = trajectory_distance_graph
        edge_attr = np.array(edge_attr)
        edge_attr = np.where(edge_attr < distance_scale, edge_attr, distance_scale)
        trajectory_distance_graph = [np.array(x, dtype=np.int64), np.array(edge_index, dtype=np.int64), np.array(edge_attr, dtype=np.float32)]
        
        time_interval_list = np.array(time_interval_list) / 3600
        time_interval_list = np.where(time_interval_list < time_interval_scale, time_interval_list, time_interval_scale)
        time_interval_list = np.array(time_interval_list, dtype=np.float32)

        time_interval_label = time_interval_label / 3600
        time_interval_label = time_interval_label if time_interval_label < time_interval_scale else time_interval_scale
        
        distance_label = distance_label if distance_label < distance_scale else distance_scale
            
        data = pickle.dumps([user_index, poi_list, dt_split_list, time_interval_list, distance_interval_list, trajectory_time_graph, trajectory_distance_graph, visited_graph, friend_visited_graph, last_visit_poi, poi_label, distance_label, time_interval_label, is_visited_label])
        new_txn.put(str(index).encode(), data)

train_env = lmdb.open(f"./{dataset}/train_dataset", readonly=True)
train_txn = train_env.begin()
train_index_list = [str(i) for i in range(train_txn.stat()["entries"])]

test_env = lmdb.open(f"./{dataset}/test_dataset", readonly=True)
test_txn = test_env.begin()
test_index_list = [str(i) for i in range(test_txn.stat()["entries"])]

new_train_env = lmdb.open(f"./{dataset}/train_dataset_trans", map_size=1099511627776)
new_train_txn = new_train_env.begin(write=True)
new_train_index_list = [str(i) for i in range(new_train_txn.stat()["entries"])]
new_test_env = lmdb.open(f"./{dataset}/test_dataset_trans", map_size=1099511627776)
new_test_txn = new_test_env.begin(write=True)
dataset_trans(train_index_list, train_txn, new_train_txn)
new_train_txn.commit()
new_train_env.close()

dataset_trans(test_index_list, test_txn, new_test_txn)
new_test_txn.commit()
new_test_env.close()

if use_scale == True:
    near_graph = np.load(f"./{dataset}/near_graph.npy", allow_pickle=True).tolist()
    last_visit_graph_dict = dict()
    for key, value in tqdm(near_graph.items()):
        x, edge_index, edge_attr = value
        edge_attr = np.array(edge_attr)
        edge_attr = np.where(edge_attr < distance_scale, edge_attr, distance_scale)
        last_visit_graph_dict[key] = [np.array(x, dtype=np.int64), np.array(edge_index, dtype=np.int64), np.array(edge_attr, dtype=np.float32)]
    np.save(f"./{dataset}/near_graph_trans.npy", arr=last_visit_graph_dict)