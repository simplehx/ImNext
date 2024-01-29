import random
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from torch_geometric.loader import DataLoader
from multiprocessing import Manager
import lmdb
import argparse
from torch.cuda.amp import autocast
from Dataset import POIDataset
from Model import POIPredictionModel
import torch_geometric.transforms as T
import torch.nn.functional as F
import datetime
from torch.cuda.amp.grad_scaler import GradScaler
from torch.cuda.amp.autocast_mode import autocast
import os
import logging

parser = argparse.ArgumentParser()
dataset = "gowalla"
parser.add_argument("--last_visit_graph_dict_path", type=str, default=f"./{dataset}/near_graph_trans.npy")
parser.add_argument("--data_path", type=str, default=f"./{dataset}/data.npz")
parser.add_argument("--train_dataset_path", type=str, default=f"./{dataset}/train_dataset_trans")
parser.add_argument("--test_dataset_path", type=str, default=f"./{dataset}/test_dataset_trans")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--grad_norm_alpha", type=float, default=0.8)
parser.add_argument("--batch_size", type=int, default=200)
parser.add_argument("--num_workers", type=int, default=5)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--seq_len", type=int, default=5)
parser.add_argument("--layer", type=float, default=4)
parser.add_argument("--dimension", type=float, default=256)
parser.add_argument("--save_path", type=str, default="./save/")


args = parser.parse_args()


class MultiTaskLossFunction(nn.Module):
    def __init__(self):
        super(MultiTaskLossFunction, self).__init__()
        self.is_visited_loss_fn = nn.CrossEntropyLoss()
        self.distance_loss_fn = nn.L1Loss()
        self.time_interval_loss_fn = nn.L1Loss()
        self.next_poi_loss_fn = nn.CrossEntropyLoss()

    def forward(self, is_visited_pred_output, is_visited_true, distance_pred_output, distance_true, time_interval_pred_output, time_interval_true, next_poi_pred_output, next_poi_true):
        is_visited_loss = self.is_visited_loss_fn(is_visited_pred_output, is_visited_true)
        distance_loss = self.distance_loss_fn(distance_pred_output, distance_true)
        time_interval_loss = self.time_interval_loss_fn(time_interval_pred_output, time_interval_true)
        next_poi_loss = self.next_poi_loss_fn(next_poi_pred_output, next_poi_true)
        return is_visited_loss, distance_loss, time_interval_loss, next_poi_loss

class AutomaticWeightedLoss(nn.Module):
    def __init__(self, num):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, main_task_loss, *x):
        loss_sum = main_task_loss
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def metrics(out, y):
    _, top1_indices = torch.topk(out, 1, dim=1)
    _, top5_indices = torch.topk(out, 5, dim=1)
    _, top10_indices = torch.topk(out, 10, dim=1)
    top1_indices = top1_indices.view(-1, 1)
    top5_indices = top5_indices.view(-1, 5)
    top10_indices = top10_indices.view(-1, 10)
    batch_total = out.size(0)
    # y = y.transpose(1, 0).reshape(-1, 1)
    rank = [torch.argwhere(torch.argsort(pred, descending=True) == target).item() + 1 for pred, target in zip(out, y.cuda())]
    reciprocal_rank = (1 / torch.tensor(rank)).tolist()
    y = y.view(-1, 1)
    top1_correct = (y == top1_indices).sum().item()
    top5_correct = (y == top5_indices).sum().item()
    top10_correct = (y == top10_indices).sum().item()

    return batch_total, top1_correct, top5_correct, top10_correct, reciprocal_rank

class Trainner:
    def __init__(self):
        self.scaler = GradScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.model = POIPredictionModel(num_classes, poi_count, user_count, year_list, args.dropout, args.seq_len, args.layer, args.dimension).to(self.device)

        self.awl = AutomaticWeightedLoss(3).to(self.device)
        self.optimizer = torch.optim.AdamW([{"params": self.model.parameters()}, {"params": self.awl.parameters()}], lr=args.learning_rate)
        self.loss_function = MultiTaskLossFunction().to(self.device)

        self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer, step_size=5, gamma=0.8)

    def metrics(self, out, y):
        _, top1_indices = torch.topk(out, 1, dim=1)
        _, top5_indices = torch.topk(out, 5, dim=1)
        _, top10_indices = torch.topk(out, 10, dim=1)
        top1_indices = top1_indices.view(-1, 1)
        top5_indices = top5_indices.view(-1, 5)
        top10_indices = top10_indices.view(-1, 10)
        batch_total = out.size(0)
        # y = y.transpose(1, 0).reshape(-1, 1)
        rank = [torch.argwhere(torch.argsort(pred, descending=True) == target).item() + 1 for pred, target in zip(out, y.cuda())]
        reciprocal_rank = (1 / torch.tensor(rank)).tolist()
        y = y.view(-1, 1)
        top1_correct = (y == top1_indices).sum().item()
        top5_correct = (y == top5_indices).sum().item()
        top10_correct = (y == top10_indices).sum().item()

        return batch_total, top1_correct, top5_correct, top10_correct, reciprocal_rank

    def iteration(self, data_loader, is_train=True):
        epoch_loss = []
        next_poi_loss_list, is_visited_loss_list, distance_loss_list, time_interval_loss_list = [], [], [], []
        epoch_total, epoch_top1_correct, epoch_top5_correct, epoch_top10_correct = 0, 0, 0, 0
        epoch_reciprocal_rank = []
        for data in tqdm(data_loader):
            data = {key: value.to(self.device) for key, value in data.items()}
            with autocast():
                is_visited_pred_output, distance_pred_output, time_interval_pred_output, next_poi_pred_output = self.model(data["user_index"], data["poi_list"], data["dt_split_list"], data["time_interval_list"], data["distance_interval_list"], data["trajectory_time_graph"], data["trajectory_distance_graph"], data["last_visit_graph"], data["visited_graph"], data["friend_visited_graph"])
                is_visited_loss, distance_loss, time_interval_loss, next_poi_loss = self.loss_function(is_visited_pred_output, data["is_visited_label"], distance_pred_output, data["distance_label"].unsqueeze(1), time_interval_pred_output, data["time_interval_label"].unsqueeze(1), next_poi_pred_output, data["poi_label"])
                loss_sum = self.awl(next_poi_loss, is_visited_loss, distance_loss, time_interval_loss)
                # loss_sum = is_visited_loss + distance_loss + time_interval_loss + next_poi_loss
                # loss_sum = time_interval_loss
                next_poi_loss_list.append(next_poi_loss.item())
                is_visited_loss_list.append(is_visited_loss.item())
                distance_loss_list.append(distance_loss.item())
                time_interval_loss_list.append(time_interval_loss.item())

                epoch_loss.append(loss_sum.item())

                if not is_train:
                    batch_total, batch_top1_correct, batch_top5_correct, batch_top10_correct, batch_reciprocal_rank = self.metrics(next_poi_pred_output, data["poi_label"])
                    epoch_total += batch_total
                    epoch_top1_correct += batch_top1_correct
                    epoch_top5_correct += batch_top5_correct
                    epoch_top10_correct += batch_top10_correct
                    epoch_reciprocal_rank.extend(batch_reciprocal_rank)
            
            if is_train:
                self.scaler.scale(loss_sum).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        

        if is_train:
            next_poi_loss, is_visited_loss, distance_loss, time_interval_loss = -1, -1, -1, -1
            if args.use_multi_task:
                next_poi_loss, is_visited_loss, distance_loss, time_interval_loss = np.mean(next_poi_loss_list), np.mean(is_visited_loss_list), np.mean(distance_loss_list), np.mean(time_interval_loss_list)
            result = {"loss": np.mean(epoch_loss), "next_poi_loss": next_poi_loss, "is_visited_loss": is_visited_loss, "distance_loss": distance_loss, "time_interval_loss": time_interval_loss}
        else:
            epoch_top1_acc = epoch_top1_correct / epoch_total * 100
            epoch_top5_acc = epoch_top5_correct / epoch_total * 100
            epoch_top10_acc = epoch_top10_correct / epoch_total * 100
            mrr = np.mean(epoch_reciprocal_rank)
            result = {"loss": np.mean(epoch_loss), "acc1": epoch_top1_acc, "acc5": epoch_top5_acc, "acc10": epoch_top10_acc, "mrr": mrr}
    
        return result

    def train(self):
        self.model.train()
        result = self.iteration(train_loader, is_train=True)
        loss, next_poi_loss, is_visited_loss, distance_loss, time_interval_loss = result["loss"], result["next_poi_loss"], result["is_visited_loss"], result["distance_loss"], result["time_interval_loss"]
        return loss, next_poi_loss, is_visited_loss, distance_loss, time_interval_loss
    
    def val(self):
        self.model.eval()
        with torch.no_grad():
            result = self.iteration(val_loader, is_train=False)
            loss, acc1, acc5, acc10, mrr = result["loss"], result["acc1"], result["acc5"], result["acc10"], result["mrr"]
        return loss, acc1, acc5, acc10, mrr

    def test(self):
        self.model.eval()
        with torch.no_grad():
            result = self.iteration(test_loader, is_train=False)
            loss, acc1, acc5, acc10, mrr = result["loss"], result["acc1"], result["acc5"], result["acc10"], result["mrr"]
        return loss, acc1, acc5, acc10, mrr

    def start(self):
        logging.info(f"layer: {args.layer}, dim: {args.dimension}, seq_len: {args.seq_len}")
        best_val_acc10 = 0
        for epoch in range(args.epochs):
            train_loss, next_poi_loss, is_visited_loss, distance_loss, time_interval_loss = self.train()
            logging.info(f"epoch: {epoch} | train loss: {train_loss} | next_poi_loss: {next_poi_loss} | is_visited_loss: {is_visited_loss} | distance_loss: {distance_loss} | time_interval_loss: {time_interval_loss}")
            val_loss, val_acc1, val_acc5, val_acc10, val_mrr = self.val()
            if val_acc10 > best_val_acc10:
                best_val_acc10 = val_acc10
                logging.info("best model")
            test_loss, test_acc1, test_acc5, test_acc10, test_mrr = self.test()
            logging.info(f"val loss: {val_loss} | acc1: {val_acc1} | acc5: {val_acc5} | acc10: {val_acc10} | mrr : {val_mrr}")
            logging.info(f"test loss: {test_loss} | acc1: {test_acc1} | acc5: {test_acc5} | acc10: {test_acc10} | mrr : {test_mrr}")
            logging.info("-" * 100)
            
def dt_converter(*args):
    now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    return now.timetuple()

if __name__ == '__main__':
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    logging_format = "%(asctime)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.Formatter.converter = dt_converter
    logging.basicConfig(filename="result.log", level=logging.INFO, format=logging_format, datefmt=date_format)


    train_data = np.load(f"./{dataset}/data.npz", allow_pickle=True)
    index2poi, index2user, year_list = train_data["index2poi"].tolist(), train_data["index2user"].tolist(), train_data["year_list"].tolist()
    num_classes, poi_count, user_count = len(index2poi), len(index2poi), len(index2user)

    last_visit_graph_dict = np.load(f"./{dataset}/near_graph_trans.npy", allow_pickle=True).tolist()

    train_env = lmdb.open(f"./{dataset}/train_dataset_trans", readonly=True)
    train_txn = train_env.begin(buffers=True)
    train_val_dataset_len = train_txn.stat()["entries"]
    train_len = int(train_val_dataset_len * 0.9)
    dataset_index_list = [str(i) for i in range(train_val_dataset_len)]

    random.shuffle(dataset_index_list)
    train_index, val_index = dataset_index_list[:train_len], dataset_index_list[train_len:]

    train_dataset = POIDataset(train_index, last_visit_graph_dict, train_txn)
    val_dataset = POIDataset(val_index, last_visit_graph_dict, train_txn)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, prefetch_factor=6)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, prefetch_factor=6)

    test_env = lmdb.open(f"./{dataset}/test_dataset_trans", readonly=True)
    test_txn = test_env.begin()
    test_dataset_len = test_txn.stat()["entries"]
    test_index_list = [str(i) for i in range(test_dataset_len)]
    test_dataset = POIDataset(test_index_list, last_visit_graph_dict, test_txn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, prefetch_factor=6)

    trainner = Trainner()
    trainner.start()
    train_env.close()
    test_env.close()