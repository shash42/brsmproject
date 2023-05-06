import pandas as pd
import random
import scipy
from tabulate import tabulate
import numpy as np
from math import sqrt
from utils import get_session_list, disallow_context_change
from matplotlib import pyplot as plt

random.seed(42)

def get_index_from_features(df):
    # first is index, second is feature
    feature_indices = [[], []]
    for i in range(len(df.keys())):
        feature_indices[0].append(i)
        feature_indices[1].append(df.keys()[i])
    return feature_indices

def get_2_tracks(session):
    #returns track ids
    track1 = df_1.loc[track_idxs[0]].values
    track2 = df_1.loc[track_idxs[1]].values
    track1, track2 = track1[3], track2[3]
    return track1, track2    

def get_track(session, track_idx):
    #returns track id
    track_id = session[track_idx][3]
    return track_id


def find_dists(sesh_pairs, ftid):
    dists = []
    for pair in sesh_pairs:
        idxs = df_2.index[df_2['track_id'] == pair[0]].tolist()
        idx1 = idxs[0]
        idxs = df_2.index[df_2['track_id'] == pair[1]].tolist()
        idx2 = idxs[0]
        dist = abs(df_features[idx1][ftid] - df_features[idx2][ftid])
        # print(idx1, idx2, dist, df_features[idx1][ftid])
        
        dists.append(dist)

    return dists

def cohend(d1, d2):
    # calculate the size of samples
    n1, n2 = len(d1), len(d2)
    # calculate the variance of the samples
    s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
    # calculate the pooled standard deviation
    s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    # calculate the means of the samples
    u1, u2 = np.mean(d1), np.mean(d2)
    # calculate the effect size
    return (u1 - u2) / s

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

num_sesh = len(session_list)
num_samples = 1000

for i in range(num_samples):
    sesh_id = random.randint(0, num_sesh-1)
    session = session_list[sesh_id]
    num_tracks = len(session)
    track_idxs = random.sample(range(num_tracks), 2)
    track1, track2 = get_track(session, track_idxs[0]), get_track(session, track_idxs[1])
    samesesh.append((track1, track2))
    idx1 = df_2.index[df_2['track_id'] == track1].tolist()[0]
    # print(df_features[idx1][2])

for i in range(num_samples):
    sesh_ids = random.sample(range(num_sesh), 2)
    # print(sesh_ids[0])
    session = session_list[sesh_ids[0]]
    track1 = get_track(session, random.randint(0, len(session) - 1))
    # print(sesh_ids[1])
    session = session_list[sesh_ids[1]]
    track2 = get_track(session, random.randint(0, len(session) - 1))
    diffsesh.append((track1, track2))

# print(samesesh, diffsesh)

normal_same, normal_diff = [], []
cohens = []
mwus, ps = [], []
means_same, means_diff = [], []
vars_same, vars_diff = [], []
fts = []
for ftid in range(1, 22):
    if ftid==16:
        continue
    print(ftid)
    fts.append(feature_indices[1][ftid])
    dists_samesesh = find_dists(samesesh, ftid)
    dists_diffsesh = find_dists(diffsesh, ftid)
    # gauss_same = scipy.stats.shapiro(dists_samesesh)
    # gauss_diff = scipy.stats.shapiro(dists_diffsesh)
    # normal_same.append(gauss_same[1])
    # normal_diff.append(gauss_diff[1])
    # plt.hist(dists_samesesh, bins=10)
    # plt.show()
    # plt.hist(dists_diffsesh, bins=10)
    # plt.show()
    mwu = scipy.stats.mannwhitneyu(dists_samesesh, dists_diffsesh, alternative='less')
    mwus.append(mwu.statistic)
    ps.append(mwu.pvalue)
    effectsz = cohend(dists_samesesh, dists_diffsesh)
    cohens.append(effectsz)

    stats_same = scipy.stats.describe(dists_samesesh)
    stats_diff = scipy.stats.describe(dists_diffsesh)
    means_same.append(stats_same.mean)
    means_diff.append(stats_diff.mean)
    vars_same.append(stats_same.variance)
    vars_diff.append(stats_diff.variance)

homotab = tabulate([means_same, means_diff, vars_same, vars_diff, ps, cohens], headers=fts, tablefmt="tsv")
text_file=open("homogenity.tsv","w")
text_file.write(homotab)
text_file.close()

