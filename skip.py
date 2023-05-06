import pandas as pd
import random
import scipy
from tabulate import tabulate
import numpy as np
from math import sqrt
from utils import get_session_list, disallow_context_change
from matplotlib import pyplot as plt
import pickle

random.seed(42)

def get_index_from_features(df):
    # first is index, second is feature
    feature_indices = [[], []]
    for i in range(len(df.keys())):
        feature_indices[0].append(i)
        feature_indices[1].append(df.keys()[i])
    return feature_indices

df_1 = pd.read_csv('data/log_mini.csv')
df_2 = pd.read_csv('data/tf_mini.csv')
feature_indices = get_index_from_features(df_2)

print(len(df_1))
# df to array
df_array = df_1.values
df_features = df_2.values
 

session_list = get_session_list(df_array)
print(len(session_list))
# session_list = disallow_context_change(session_list)

samesesh = []
diffsesh = []

def get_idx(trackid):
    idxs = df_2.index[df_2['track_id'] == trackid].tolist()
    return idxs[0]


skipdiffs = []

off_skip = 4


for ftid in range(1, 5):
    skipdiffs.append([[], [], [], []])
    if ftid==16:
        continue
    for idx, session in enumerate(session_list):
        if idx % 10 == 0:
            print(idx)
        if idx >= 100:
            break
        
        sesh_vals = []
        for track in session:
            idx = get_idx(track[3])
            sesh_vals.append(df_features[idx][ftid])
        mean_val = sum(sesh_vals) / len(sesh_vals)

        for track in session:
            for j in range(4, 8):
                if track[j] == True:
                    trackj_val = df_features[get_idx(track[3])][ftid]
                    skipdiffs[-1][j - off_skip].append(trackj_val - mean_val)

fts = []
means = [[], [], [], []]
idx = -1
for ftid in range(1, 22):
    if ftid == 16:
        continue
    fts.append(feature_indices[1][ftid])
    idx += 1
    for j in range(4):
        print(fts[idx])
        print(j, scipy.stats.describe(skipdiffs[idx][j]).mean)
        # means[j].append(scipy.stats.describe(skipdiffs[idx][j]).mean)

# from tabulate import tabulate
# homotab = tabulate([means[0], means[1], means[2], means[3]], headers=fts, tablefmt="tsv")
# text_file=open("skips.tsv","w")
# text_file.write(homotab)
# text_file.close()

