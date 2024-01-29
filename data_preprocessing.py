import random
import numpy as np
from tqdm import tqdm
import pandas as pd
from tqdm import tqdm
from haversine import haversine, Unit
import os
from haversine import haversine_vector, Unit
import multiprocessing
from multiprocessing import Manager
from collections import Counter

poi_min = 20
traj_len_min = 20

dataset = "gowalla"
if not os.path.exists(dataset):
    os.makedirs(dataset)

def get_check_in_df_raw():
    file_path=f"{dataset}/check_in_df_raw.npy"
    if not os.path.exists(file_path):
        df = pd.read_table(f"./raw_data/checkins-{dataset}.txt", names=["user", "dt", "lat", "lng", "poi"])
        df["dt"] = df["dt"].astype('datetime64[s]')
        poi_list = df["poi"].value_counts()[df["poi"].value_counts() > poi_min].index.tolist()
        df = df.set_index("poi").loc[poi_list].sort_values("dt").reset_index()
        check_in_df = df.to_dict("list")
        # location_df = df.drop_duplicates("poi")[["poi", "lat", "lng"]].to_dict("list")
        np.save(file_path, arr=check_in_df)
    else:
        check_in_df = np.load(file_path, allow_pickle=True).tolist()
    check_in_df = pd.DataFrame(check_in_df)
    return check_in_df

def get_trajectory():
    file_path=f"{dataset}/trajectory.npz"
    if not os.path.exists(file_path):
        check_in_df_raw = get_check_in_df_raw()
        # filter data
        for i in range(1):
            poi_count = check_in_df_raw.value_counts("poi")
            poi_list = poi_count[poi_count > poi_min].index.tolist()
            check_in_df_raw = check_in_df_raw[check_in_df_raw["poi"].isin(poi_list)]
            user_traj_count = check_in_df_raw.groupby("user")["poi"].count()
            user_list = user_traj_count[user_traj_count > traj_len_min].index.tolist()
            check_in_df_raw = check_in_df_raw[check_in_df_raw["user"].isin(user_list)]
        
        all_poi, all_user = check_in_df_raw["poi"].drop_duplicates().tolist(), check_in_df_raw["user"].drop_duplicates().tolist()
        index2poi = {index: poi for index, poi in enumerate(all_poi)}
        poi2index = {poi: index for index, poi in enumerate(all_poi)}
        index2user = {index: user for index, user in enumerate(all_user)}
        user2index = {user: index for index, user in enumerate(all_user)}

        user_trajectory = dict()
        check_in_df_raw["poi"] = check_in_df_raw["poi"].apply(lambda x: poi2index[x])
        check_in_df_raw["user"] = check_in_df_raw["user"].apply(lambda x: user2index[x])
        for user, user_value_df in tqdm(check_in_df_raw.groupby("user")):
            user_poi_list = user_value_df["poi"].tolist()
            user_dt_list = user_value_df["dt"].tolist()
            user_trajectory[user] = [user_poi_list, user_dt_list]
        year_list = pd.unique(check_in_df_raw["dt"].apply(lambda x: str(x)[:4])).astype(int).tolist()
        np.savez(file_path, user_trajectory=user_trajectory, all_user=all_user, all_poi=all_poi, index2poi=index2poi, poi2index=poi2index, index2user=index2user, user2index=user2index, year_list=year_list)
        
        location_df = check_in_df_raw.drop_duplicates("poi")[["poi", "lat", "lng"]]
        np.save(f"{dataset}/location_df.npy", arr=location_df.to_dict("list"))
    else:
        file = np.load(file_path, allow_pickle=True)
        user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list = file["user_trajectory"], file["all_user"], file["all_poi"], file["index2poi"], file["poi2index"], file["index2user"], file["user2index"], file["year_list"]
        user_trajectory = user_trajectory.tolist()
        index2poi, poi2index = index2poi.tolist(), poi2index.tolist()
        index2user, user2index = index2user.tolist(), user2index.tolist()
        year_list = year_list.tolist()
    return user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list

def get_friends_dict():
    file_path = f"{dataset}/friends.npy"
    if not os.path.exists(file_path):
        user_df = pd.read_table(f"./raw_data/{dataset}_friend.txt", names=["from", "to"])
        user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list = get_trajectory()
        user_df = user_df[user_df["from"].isin(all_user) & user_df["to"].isin(all_user)]
        
        friends_dict = dict()
        for user in tqdm(all_user):
            friends_dict[user2index[user]] = user_df[user_df["from"] == user]["to"].apply(lambda x: user2index[x]).tolist()
        np.save(file_path, arr=friends_dict)
    else:
        friends_dict = np.load(file_path, allow_pickle=True).tolist()
    return friends_dict

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

def get_near_graph_subtask(poi_index_split, namespace, padding_index, index):
    print(index, "正在处理")
    poi_location_df = namespace.poi_location_df
    check_in_df = namespace.check_in_df

    visit_count = check_in_df["poi"].value_counts()
    check_in_df = check_in_df.set_index("poi")
    poi_location_df_poi_index = poi_location_df.set_index("poi")
    result = []
    for poi_index in poi_index_split:
        from_lat, from_lng = poi_location_df[poi_location_df["poi"] == poi_index].iloc[0][["lat", "lng"]]
        # 缩小搜索范围
        lat_min, lat_max = from_lat - 0.1, from_lat + 0.1
        lng_min, lng_max = from_lng - 0.1, from_lng + 0.1
        # a->b == b->a 只保留一个
        near_df = poi_location_df[(poi_index < poi_location_df["poi"]) & (poi_location_df["lat"].between(lat_min, lat_max)) & (poi_location_df["lng"].between(lng_min, lng_max))]
        near_location = near_df[["lat", "lng"]].values.tolist()
        poi_near_result = [[padding_index], [], [], []]
        if len(near_location) > 0:
            near_distance = haversine_vector([from_lat, from_lng], near_location, Unit.KILOMETERS, comb=True)[:,0].tolist() # 计算poi与附近poi的距离
            near_df.insert(0, "distance", near_distance)
            near_df = near_df[near_df["distance"] < 5] # 小于5km
            near_df["visit_count"] = visit_count.loc[near_df["poi"]].values
            near_df = near_df.sort_values("visit_count", ascending=False)[:50] # 5km内的50个热门poi
            near_df["from"] = poi_index
            # 构造图
            near_poi_unique = np.unique([near_df["poi"], near_df["from"]]) # 需要计算的所有POI
            all_user_traj = check_in_df.loc[near_poi_unique].sort_values("user").index.values
            pair_list = [all_user_traj[i: i + 2] for i in range(len(all_user_traj) - 1)]
            pair_list.extend(near_df[["from", "poi"]].values.tolist())
            if len(pair_list) > 0:
                user_traj_df = pd.DataFrame(np.unique(pair_list, axis=0), columns=["from", "to"])
                user_traj_df = user_traj_df.assign(temp=user_traj_df.apply(lambda x: tuple(sorted(x)), axis=1)).drop_duplicates('temp')
                poi_index_trans_df = pd.DataFrame(np.arange(len(near_poi_unique)), index=near_poi_unique, columns=["index"])
                x = poi_index_trans_df.index.tolist()
                from_poi = poi_index_trans_df.loc[user_traj_df["from"]]["index"].tolist()
                to_poi = poi_index_trans_df.loc[user_traj_df["to"]]["index"].tolist()
                from_location = poi_location_df_poi_index.loc[user_traj_df["from"]].values
                to_location = poi_location_df_poi_index.loc[user_traj_df["to"]].values
                edge_attr = haversine_vector(from_location, to_location, Unit.KILOMETERS).tolist()
                if len(x) > 0:
                    poi_near_result = [x, from_poi, to_poi, edge_attr]
        x, from_poi, to_poi, edge_attr = poi_near_result
        from_poi, to_poi, edge_attr = trans2undirectedGraph(from_poi, to_poi, edge_attr, method="mean")
        poi_near_result = [x, [from_poi, to_poi], edge_attr]
        result.append([poi_index, poi_near_result])
    return result

# 如需在训练时生成，可以使用Ball-Tree优化
def get_near_graph():
    file_path = f"{dataset}/near_graph.npy"
    if not os.path.exists(file_path):
        near_graph = dict()
        user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list = get_trajectory()
        padding_index = len(poi2index)
        poi_location_df = np.load(f"{dataset}/location_df.npy", allow_pickle=True).tolist()
        poi_location_df = pd.DataFrame(poi_location_df)
        check_in_df = get_check_in_df()
        namespace = Manager().Namespace()
        namespace.poi_location_df = poi_location_df
        namespace.check_in_df = check_in_df

        processes = 15
        all_poi_index = list(poi2index.values())
        random.shuffle(all_poi_index)
        all_poi_index_split = np.array_split(all_poi_index, processes)
        pbar = tqdm(total=len(all_poi_index_split))
        pbar_update = lambda *args: pbar.update()
        result_list = []

        pool = multiprocessing.Pool(processes=processes)
        for index, poi_index_split in enumerate(all_poi_index_split):
            result = pool.apply_async(get_near_graph_subtask, (poi_index_split, namespace, padding_index, index), callback=pbar_update)
            result_list.append(result)
            print(index, "已提交")
        pool.close()
        pool.join()

        for result in result_list:
            for poi, poi_near_result in result.get():
                near_graph[poi] = poi_near_result
        np.save(file_path, arr=near_graph)
    else:
        near_graph = np.load(file_path, allow_pickle=True).tolist()
    return near_graph

def get_check_in_df():
    file_path=f"{dataset}/check_in_df.npy"
    if not os.path.exists(file_path):
        check_in_df_raw = get_check_in_df_raw()
        user_trajectory, all_user, all_poi, index2poi, poi2index, index2user, user2index, year_list = get_trajectory()
        check_in_df = check_in_df_raw[(check_in_df_raw["poi"].isin(all_poi)) & (check_in_df_raw["user"].isin(all_user))]
        check_in_df["poi"] = check_in_df["poi"].apply(lambda x: poi2index[x]).tolist()
        check_in_df["user"] = check_in_df["user"].apply(lambda x: user2index[x]).tolist()
        np.save(file_path, arr=check_in_df[["user", "poi", "dt"]].to_dict("list"))
    else:
        check_in_df = np.load(file_path, allow_pickle=True).tolist()
    check_in_df = pd.DataFrame(check_in_df)
    return check_in_df



if __name__ == '__main__':
    get_check_in_df_raw()
    get_trajectory()
    get_friends_dict()
    get_check_in_df()
    get_near_graph()