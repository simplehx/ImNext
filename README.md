# ImNext: Irregular Interval Attention and Multi-task Learning for Next POI Recommendation

The next point-of-interest (POI) recommendation task recommends users POIs that they may be interested in based on their historical trajectories. This task holds value for users as well as service providers; however, it is difficult. Although users exhibit repetitive and periodic behavioral characteristics at the macro level, such characteristics are influenced by individual preferences and diverse factors at the micro level, rendering prediction difficult. Most existing sequence modeling methods consider intervals between elements to be invariants. However, the time and distance intervals between adjacent POIs in the user's check-in sequences are irregular, which contains significant user behavioral characteristics. Therefore, we propose a model known as **I**rregular Interval Attention and **M**ulti-task Learning for **Next** POI Recommendation (ImNext). First, to address data sparsity and irregular intervals in the check-in sequence, we designed a data augmentation method to improve data density and proposed a novel irregular interval attention (IrrAttention) module. Second, to deal with the potential factors that affect user behavior, we proposed a graph attention network module that integrates edge attention (EA-GAT), which incorporates edge weights in the user's spatiotemporal and social transition graphs. Lastly, we established multiple subtasks for joint learning as the user’s next check-in hides multiple targets, such as time and distance intervals. The experimental results show that our proposed method outperforms the state-of-the-art (SOTA) methods on two real-world public datasets.


## Requirements
* Python == 3.10.12
* tqdm == 4.65.0
* numpy == 1.25.2
* pandas == 2.0.3
* pytorch == 2.0.1
* lmdb == 1.4.1
* torch_geometric == 2.3.1
* pytorch-scatter == 2.1.1
* pytorch-sparse == 0.6.17

## Datasets
The original dataset can be downloaded from:
* Gowalla: https://snap.stanford.edu/data/loc-gowalla.html
* Foursquare: https://sites.google.com/site/yangdingqi/home/foursquare-dataset

## Usage
* step 1: download dataset to ./raw_data
* step 2: run data_preprocessing.py
* step 3: run generate_dataset.py
* step 4: run dataset_trans.py
* step 5: run train.py


## Reference
```
@article{he2024imnext,
  title={ImNext: Irregular Interval Attention and Multi-Task Learning for Next POI Recommendation},
  author={He, Xi and He, Weikang and Liu, Yilin and Lu, Xingyu and Xiao, Yunpeng and Liu, Yanbing},
  journal={Knowledge-Based Systems},
  pages={111674},
  year={2024},
  publisher={Elsevier}
}
```
