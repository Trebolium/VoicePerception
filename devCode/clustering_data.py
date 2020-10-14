#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:11:45 2020

@author: brendanoconnor
"""

import util, h5py, os
import numpy as np
from time import time
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
import math, itertools
import scipy.stats as stats
from kneed import KneeLocator
import pandas as pd
import seaborn as sns


if 'h5' in globals():
    h5.close()
h5 = h5py.File('/Users/brendanoconnor/Desktop/studyData20200707.hdf5', mode = 'r')


num_experiments = len(h5['pageInfo'])
all_indices = np.arange(num_experiments)
label_list = ['straight','belt','breathy','fry','vibrato']
show_plot = False

# GENERATE TWO LISTS OF BEST K VALUES USING ELBOW AND SILHOUETTE TECHNIQUES
all_elbows = util.get_elbow_array(h5, all_indices, range(1,14), 'all_elbows', show_plot, 0.25)
all_ks_from_silhouettes = util.get_k_array_from_silhouettes(h5, all_indices, range(2,14), 'all_ks_from_silhouettes', True)

#LOOKS OVER EACH PARTICIPANT'S RATINGS, ASSESSING THE SCORE OF ELBOW AND SILHOUETTE (ARRANGED BY SUBGROUP)
elbow_groups=[]
ks_from_silhouette_groups=[]
indices_groups_list_names = ['sessions_indices_lists', 'singers_indices_lists', 'registers_indices_lists', 'genders_indices_lists']
for grp_idx, group_indices in enumerate(subgroups_indices_list):
    elbow_condition_list = []
    silhouette_condition_list = []
    for con_idx, condition_indices in enumerate(group_indices):
        ks_from_silhouettes = util.get_k_array_from_silhouettes(h5, condition_indices, range(2,14), indices_groups_list_names[grp_idx]+' - Condition '+str(con_idx), show_plot)
        elbows = util.get_elbow_array(h5, condition_indices, range(1,14), indices_groups_list_names[grp_idx]+' - Condition '+str(con_idx), show_plot, 1.0)
        silhouette_condition_list.append(ks_from_silhouettes)
        elbow_condition_list.append(elbows)
    elbow_groups.append(elbow_condition_list)
    ks_from_silhouette_groups.append(silhouette_condition_list)

# LOOKS OVER EACH PARTICIPANT AND GENERATES METRIC SCORES FOR THEIR CLUSTERING BEHAVIOUR
cluster_perf_2 = np.empty((0,6))
cluster_perf_5 = np.empty((0,6))
for idx in range(num_experiments):
    for best_num_clusters in [2,5]:
    #    print(idx, h5['participantInfo'][idx])
        rearranged_dissim_matrix = h5['rearrangedDissimMatrix'][idx]
        rearranged_ref_audio = h5['rearrangedReferenceAudioNames'][idx]
        rearranged_class_list = util.strings_to_classes(rearranged_ref_audio, label_list)
    #    dissim_matrix = np.around(rearranged_dissim_matrix, 2)
        if show_plot == True:
            util.display_dissim_heatmaps(rearranged_dissim_matrix, rearranged_ref_audio)
        util.mds_from_dissim_matrix(rearranged_dissim_matrix, rearranged_ref_audio, 3, show_plot=show_plot)
        # CLUSTER
        # dendrogram = sch.dendrogram(sch.linkage(rearranged_dissim_matrix, method='ward'))
        hc = AgglomerativeClustering(n_clusters=best_num_clusters, affinity = 'euclidean', linkage='ward')
        hc_clustering = hc.fit_predict(rearranged_dissim_matrix)
        # the silhouette score is higher when clusters are dense - bounded between -1 for incorrect clustering and
        # 1 for dense clustering. Around zero when clusters are overlapping
        hc_sil_score = silhouette_score(rearranged_dissim_matrix, hc_clustering, metric='euclidean')   
        km = KMeans(n_clusters=best_num_clusters)
        km.fit_predict(rearranged_dissim_matrix)
        sse = km.inertia_
        kmeans_refs_and_clusters = util.assign_clusters(rearranged_dissim_matrix, rearranged_ref_audio, best_num_clusters)
        kmeans_cluster_list = kmeans_refs_and_clusters[:,1].tolist()
        kmeans_cluster_accuracy = adjusted_rand_score(kmeans_cluster_list, rearranged_class_list)
        hc_cluster_accuracy = adjusted_rand_score(hc_clustering, rearranged_class_list)
        kmeans_hc_accuracy_comparison = adjusted_rand_score(hc_clustering, kmeans_cluster_list)
        if best_num_clusters == 2:
            cluster_perf_2 = np.append(cluster_perf_2, np.array([[idx, sse, kmeans_cluster_accuracy,
                                               hc_cluster_accuracy, hc_sil_score, kmeans_hc_accuracy_comparison]]), axis=0)
        else:
            cluster_perf_5 = np.append(cluster_perf_5, np.array([[idx, sse, kmeans_cluster_accuracy,
                                               hc_cluster_accuracy, hc_sil_score, kmeans_hc_accuracy_comparison]]), axis=0)
    
##################################################################################################################################
#REFORMAT GROUPINGS OF LISTS FOR STATISTICAL TESTS

num_variables = len(cluster_perf_2[0])
cluster2_scores_list = []
for conditions_indices_list in subgroups_indices_list:
    variables_list = []
    for variable_col in range(1, num_variables):
        values_list = []
        for idx in conditions_indices_list:
            values_list.append(cluster_perf_2[idx,variable_col])
        variables_list.append(values_list)
    cluster2_scores_list.append(variables_list)
    
cluster5_scores_list = []
for conditions_indices_list in subgroups_indices_list:
    variables_list = []
    for variable_col in range(1, num_variables):
        values_list = []
        for idx in conditions_indices_list:
            values_list.append(cluster_perf_5[idx,variable_col])
        variables_list.append(values_list)
    cluster5_scores_list.append(variables_list)
##################################################################################################################################
## JUST COMPARING HC ACC/SIL BETWEEN THE TWO CLUSTER SOLUTIONS
#hc_acc_mean_2 = np.mean(cluster_perf_2[:,3])
#hc_sil_mean_2 = np.mean(cluster_perf_2[:,4])
#hc_acc_mean_5 = np.mean(cluster_perf_5[:,3])
#hc_sil_mean_5 = np.mean(cluster_perf_5[:,4])
#print(hc_acc_mean_2, hc_acc_mean_5)
#print(hc_sil_mean_2, hc_sil_mean_5)
#
#asd = [hc_acc_mean_2, hc_acc_mean_5, hc_sil_mean_2, hc_sil_mean_5]
#
#for thing in asd:
#    plt.hist(thing, bins=10)
#    plt.show()
#    plt.close()
    
##################################################################################################################################
# CHECH FOR OUTLIERS AMONG PARTICIPANTS' DISSIMILARITY MATRICES
flattened_matrices = np.empty((0,225))
for idx in range(num_experiments):
    rearranged_dissim_matrix = h5['dissimMatrix15Dims'][idx]
    flattened_dissim_matrix = rearranged_dissim_matrix.flatten()
    flattened_dissim_matrix = flattened_dissim_matrix.reshape(-1, len(flattened_dissim_matrix))
    flattened_matrices = np.append(flattened_matrices,flattened_dissim_matrix, axis=0)
#convert to pandas dataframe
df = pd.DataFrame(flattened_matrices)
# swap axes
df_tran = df.T
# generate coorelation matrix for all columns
df_tran_corr = df_tran.corr('spearman')
if show_plot==True:
    sns.heatmap(df_tran_corr, cmap = "RdBu")
# perform hc clustering to find outlier clusters that are behaving abnormally
hcs = []
for i in range(1,10):
    new_hc = AgglomerativeClustering(n_clusters=i, affinity = 'euclidean', linkage='ward')
    hc_new_clustering = new_hc.fit_predict(df_tran_corr)
    print(i, hc_new_clustering)
    hcs.append(hc_new_clustering)
    
################################################################################################################################## 