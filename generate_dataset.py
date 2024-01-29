import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Manager
import pickle
import lmdb
from operator import itemgetter
from haversine import haversine, haversine_vector, Unit
from datetime import datetime
import random

dataset = "gowalla"

def get_trajectory():
    file_path = f"./{dataset}/trajectory.npz"
    file = np.load(file_path, allow_pickle=True)
    user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list = file["user_trajectory"], file["all_user"], file["all_poi"], file["index2poi"], file["poi2index"], file["index2user"], file["user2index"], file["year_list"]
    user_trajectory = user_trajectory.tolist()
    index2poi, poi2index = index2poi.tolist(), poi2index.tolist()
    index2user, user2index = index2user.tolist(), user2index.tolist()
    year_list = year_list.tolist()
    return user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list

def get_friends_dict():
    file_path = f"./{dataset}/friends.npy"
    friends_dict = np.load(file_path, allow_pickle=True).tolist()
    return friends_dict

def get_check_in_df():
    file_path = f"./{dataset}/check_in_df.npy"
    check_in_df = np.load(file_path, allow_pickle=True).tolist()
    check_in_df = pd.DataFrame(check_in_df)
    return check_in_df

def get_poi_location_df():
    file_path = f"./{dataset}/location_df.npy"
    poi_location_df = np.load(file_path, allow_pickle=True).tolist()
    poi_location_df = pd.DataFrame(poi_location_df)
    poi_location_df = poi_location_df.set_index("poi")
    return poi_location_df

def trans2directedGraph(from_list, to_list, attr_list, method):
    pair_dict = dict()
    for from_, to_, attr in zip(from_list, to_list, attr_list):
        pair = (from_, to_)
        if pair not in pair_dict:
            pair_dict[pair] = []
        pair_dict[pair].append(attr)
    new_from_list, new_to_list, new_attr_list = [], [], []
    for (from_, to_), attr in pair_dict.items():
        if method == "mean":
            attr_value = np.mean(attr)
        elif method == "sum":
            attr_value = np.sum(attr)
        new_from_list.append(from_)
        new_to_list.append(to_)
        new_attr_list.append(attr_value)
    return new_from_list, new_to_list, new_attr_list

def trans2undirectedGraph(from_list, to_list, attr_list, method):
    pair_dict = dict()
    for from_, to_, attr in zip(from_list, to_list, attr_list):
        pair = (from_, to_) if from_ < to_ else (to_, from_)
        if pair not in pair_dict:
            pair_dict[pair] = []
        pair_dict[pair].append(attr)
    new_from_list, new_to_list, new_attr_list = [], [], []
    for (from_, to_), attr in pair_dict.items():
        if method == "mean":
            attr_value = np.mean(attr)
        elif method == "sum":
            attr_value = np.sum(attr)
        new_from_list.extend([from_, to_])
        new_to_list.extend([to_, from_])
        new_attr_list.extend([attr_value, attr_value])
    return new_from_list, new_to_list, new_attr_list

def generate_dataset(user_index_split, namespace, index, padding_index, split_type="sequence"):
    user_trajectory = namespace.user_trajectory
    friends_dict = namespace.friends_dict
    check_in_df = namespace.check_in_df
    poi_location_df = namespace.poi_location_df
    result = []
    print(index, "正在处理")
    window = 11
    for process_index, user_index in enumerate(user_index_split):
        # test_start_dt = datetime.now()
        if process_index == int(len(user_index_split) * 0.25):
            print(index, "已处理25%")
        elif process_index == int(len(user_index_split) * 0.5):
            print(index, "已处理50%")
        elif process_index == int(len(user_index_split) * 0.75):
            print(index, "已处理75%")
        poi_index_list, dt_list = user_trajectory[user_index]
        # poi_index_list = list(itemgetter(*poi_list)(poi2index))
        # poi_index_list = [poi2index[poi] for poi in poi_list]
        poi_index_window_list = [poi_index_list[i: i + window] for i in range(len(poi_index_list) - window + 1)]
        dt_window_list = [dt_list[i: i + window] for i in range(len(dt_list) - window + 1)]
        user_data = []
        # friends
        user_start_dt, user_end_dt = dt_list[0], dt_list[-1]
        friends = friends_dict[user_index]
        user_check_in_df = check_in_df[(check_in_df["user"].isin(friends)) & (check_in_df["dt"].between(user_start_dt, user_end_dt))].set_index("dt")
        # near
        # poi_list_unique = np.unique(poi_index_list)
        # user_near_df = near_poi[(near_poi["from"].isin(poi_list_unique)) | (near_poi["to"].isin(poi_list_unique))]
        # s1, s2 = [], []
        for window_index, (poi_window, dt_window) in enumerate(zip(poi_index_window_list, dt_window_list)):
            df = pd.DataFrame({"poi":poi_window, "dt":dt_window})
            train, label = df.iloc[:-1], df.iloc[-1:]
            # trajectory time graph
            train_poi_unique = np.unique(train["poi"])
            poi_index_trans_df = pd.DataFrame(np.arange(len(train_poi_unique)), index=train_poi_unique, columns=["index"])
            x = poi_index_trans_df.index.tolist()
            poi_new_index = poi_index_trans_df.loc[train["poi"]]["index"].tolist()
            from_poi, to_poi = poi_new_index[:-1], poi_new_index[1:]
            trajectory_time_graph_edge_attr = np.diff(train["dt"].values.astype('datetime64[s]')).astype(int).tolist()
            train_time_interval_list = trajectory_time_graph_edge_attr
            from_poi, to_poi, trajectory_time_graph_edge_attr = trans2directedGraph(from_poi, to_poi, trajectory_time_graph_edge_attr, method="mean")
            trajectory_time_graph = [x, [from_poi, to_poi], trajectory_time_graph_edge_attr]
            # trajectory distance graph
            location_df = poi_location_df.loc[train["poi"]]
            trajectory_distance_graph_edge_attr = haversine_vector(location_df[:-1], location_df[1:], Unit.KILOMETERS).tolist()
            train_distance_interval_list = trajectory_distance_graph_edge_attr
            from_poi, to_poi, trajectory_distance_graph_edge_attr = trans2undirectedGraph(from_poi, to_poi, trajectory_distance_graph_edge_attr, method="mean")
            trajectory_distance_graph = [x, [from_poi, to_poi], trajectory_distance_graph_edge_attr]
            # visited graph
            visited_poi = poi_index_list[: window_index + 10]
            last_100_visited_poi = visited_poi[-100:]
            visited_poi_unique = np.unique(last_100_visited_poi)
            poi_index_trans_df = pd.DataFrame(np.arange(len(visited_poi_unique)), index=visited_poi_unique, columns=["index"])
            x = poi_index_trans_df.index.tolist()
            poi_new_index = poi_index_trans_df.loc[last_100_visited_poi]["index"].tolist()
            from_poi, to_poi = poi_new_index[:-1], poi_new_index[1:]
            edge_attr = [1 for i in range(len(from_poi))]
            from_poi, to_poi, edge_attr = trans2directedGraph(from_poi, to_poi, edge_attr, method="sum")
            visited_graph = [x, [from_poi, to_poi], edge_attr]
            # friend visited graph
            friend_visited_graph = [[padding_index], [], [], []]
            if len(user_check_in_df) > 0:
                start_dt, end_dt = train["dt"].iloc[0], train["dt"].iloc[-1]
                friends_checkin_df = user_check_in_df[start_dt: end_dt] # 在时间窗口内的好友访问
                visited = np.unique(visited_poi).tolist()
                friends_checkin_df = friends_checkin_df[~friends_checkin_df["poi"].isin(visited)]
                top_visite_poi = friends_checkin_df.value_counts("poi").index.tolist()[:100] # 从未访问过，但是朋友最常访问的100个poi
                friends_checkin_seq = friends_checkin_df[friends_checkin_df["poi"].isin(top_visite_poi)].reset_index().sort_values("user")["poi"].tolist()
                all_pair = [friends_checkin_seq[i: i + 2] for i in range(len(friends_checkin_seq) - 1)]
                if len(all_pair) > 0:
                    all_pair_str = ["_".join(map(str, pair)) for pair in all_pair]
                    value_count = pd.Series(all_pair_str).value_counts()
                    friends_poi_unique = np.unique(all_pair)
                    poi_index_trans_df = pd.DataFrame(np.arange(len(friends_poi_unique)), index=friends_poi_unique, columns=["index"])
                    x = poi_index_trans_df.index.tolist()
                    from_poi, to_poi = poi_index_trans_df.loc[np.array(all_pair)[:, 0]]["index"].tolist(), poi_index_trans_df.loc[np.array(all_pair)[:, 1]]["index"].tolist()
                    edge_attr = value_count.loc[all_pair_str].tolist()
                friend_visited_graph = [x, from_poi, to_poi, edge_attr]
                
            x, from_poi, to_poi, edge_attr = friend_visited_graph
            from_poi, to_poi, edge_attr = trans2directedGraph(from_poi, to_poi, edge_attr, method="sum")
            friend_visited_graph = [x, [from_poi, to_poi], edge_attr]
            # last visit near graph
            last_visit_poi = train.iloc[-1]["poi"]
            train_poi_list = train["poi"].values.tolist()
            train_dt_split_list = [[d.year, d.month, d.day, d.weekday(), d.hour, d.minute, d.second] for d in train["dt"]]
            from_location, to_location = poi_location_df.loc[df.iloc[-2:]["poi"]].values
            distance_label = haversine(from_location, to_location, unit=Unit.KILOMETERS)
            time_interval_label = np.diff(df.iloc[-2:]["dt"].values.astype('datetime64[s]')).astype(int)[0]
            poi_label = label["poi"].iloc[0]
            is_visited_label = 1 if poi_label in visited_poi_unique else 0
            user_data.append([user_index, train_poi_list, train_dt_split_list, train_time_interval_list, train_distance_interval_list, trajectory_time_graph, trajectory_distance_graph, visited_graph, friend_visited_graph, last_visit_poi, poi_label, distance_label, time_interval_label, is_visited_label])

        user_train_len = int(len(user_data) * 0.8)
        user_train_data, user_test_data = user_data[:user_train_len], user_data[user_train_len:]        
        result.append((user_train_data, user_test_data))
        # test_end_dt = datetime.now()
        # print(user, "耗时", (test_end_dt - test_start_dt).total_seconds(), sum(s1), sum(s2))

    print(index, "处理完成")
    return result


if __name__ == '__main__':
    user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list = get_trajectory()
    np.savez(f"./{dataset}/data.npz", index2poi=index2poi, index2user=index2user, year_list=year_list)
    
    friends_dict = get_friends_dict()
    check_in_df = get_check_in_df()
    poi_location_df = get_poi_location_df()
    padding_index = len(all_poi)
    # near_poi = get_near_poi()
    print("数据加载完成")

    manager = Manager()
    namespace = manager.Namespace()
    namespace.user_trajectory = user_trajectory
    namespace.friends_dict = friends_dict
    namespace.check_in_df = check_in_df
    namespace.poi_location_df = poi_location_df
    print("数据处理完成")

    processes = 16
    all_user_index = list(user2index.values())
    random.shuffle(all_user_index)
    all_user_index_split = np.array_split(all_user_index, processes)
    pbar = tqdm(total=len(all_user_index_split))
    pbar_update = lambda *args: pbar.update()
    result_list = []

    pool = multiprocessing.Pool(processes=processes)
    for index, user_index_split in enumerate(all_user_index_split):
        result = pool.apply_async(generate_dataset, (user_index_split, namespace, index, padding_index, "sequence"), callback=pbar_update)
        result_list.append(result)
        print(index, "已提交")
    pool.close()
    pool.join()

    train_dataset, test_dataset = [], []
    for result in result_list:
        for user_train_data, user_test_data in result.get():
            train_dataset.extend(user_train_data)
            test_dataset.extend(user_test_data)
    
    train_env = lmdb.open(f"./{dataset}/train_dataset", map_size=1099511627776)
    train_txn = train_env.begin(write=True)
    for index, data in enumerate(tqdm(train_dataset)):
        train_txn.put(str(index).encode(), pickle.dumps(data))
    train_txn.commit()
    train_env.close()

    test_env = lmdb.open(f"./{dataset}/test_dataset", map_size=1099511627776)
    test_txn = test_env.begin(write=True)
    for index, data in enumerate(tqdm(test_dataset)):
        test_txn.put(str(index).encode(), pickle.dumps(data))
    test_txn.commit()
    test_env.close()