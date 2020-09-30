#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 23:46:09 2020

@author: brendanoconnor
"""

import util, h5py, os, sys, math, itertools, matplotlib
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from statistics import mean, variance
from time import time

#######################################################################################
# GENERATE NUMPY WITH CLUSTERING AND PROFILE METRICS
profile_clustering_measures = np.concatenate((profile_array, cluster_perf_5), axis=1)

#REMOVE ALL UNNECESSARY FEATURES/METRICS
profile_clustering_measures_reduced = np.delete(profile_clustering_measures, 13, 1) # delete hc vrs kmeans solution similarity
profile_clustering_measures_reduced = np.delete(profile_clustering_measures_reduced, 10, 1) # delete kmeans acc as we're using hc accuracy
profile_clustering_measures_reduced = np.delete(profile_clustering_measures_reduced, 8, 1) # delete indices column that came from cluster_perf_5
profile_clustering_measures_reduced = np.delete(profile_clustering_measures_reduced, 0, 1) # delete indices column that came from profile_array

# CREATE VERSION OF CLUSTER-PROFILE FEATURES WHERE PARTICIPANTS WITH INCONCLUSIVE FEEDBACK (VALUE=5) ARE REMOVED
no_task_feedback_indices = np.where(profile_clustering_measures_reduced[:,4]==5)[0]
no_task_feedback_indices[::-1].sort() # reverses the numpy
profile_clustering_measures_reduced_shortened = profile_clustering_measures_reduced.copy()
for i in no_task_feedback_indices:
    profile_clustering_measures_reduced_shortened = np.delete(profile_clustering_measures_reduced_shortened, i, 0)

#np.set_printoptions(suppress=True)

# CORRELATION BETWEEN CLUSTER AND PROFILE METRICS
# only consider entries from coorelation_sig_array (metric_a, metric_b, r_value, p_value) that are ordinal-based data
correlation_array, correlation_sig_array = util.array_column_correlations(profile_clustering_measures_reduced, 0.05, 'spearmanr', show_plots=False)
correlation_array, correlation_sig_array = util.array_column_correlations(profile_clustering_measures_reduced_shortened, 0.05, 'spearmanr', show_plots=False)

# only consider entries from coorelation_sig_array (metric_a, metric_b, r_value, p_value) that are interval-based data
correlation_array, correlation_sig_array = util.array_column_correlations(profile_clustering_measures_reduced, 0.05, 'pearsonr', show_plots=False)

#######################################################################################

sig_thresh = 0.05
two_tailed_sig_thresh = sig_thresh/2
# reports differences between conditions for cluster metrics, class-pair distances
annotated_results, results = util.nonpara_multiple_variables_test(cluster2_scores_list, two_tailed_sig_thresh)
annotated_results, results = util.nonpara_multiple_variables_test(cluster5_scores_list, two_tailed_sig_thresh)
annotated_results, results = util.nonpara_multiple_variables_test(averaged_groups, two_tailed_sig_thresh)

#######################################################################################

# estimate sample size via power analysis
from statsmodels.stats.power import TTestIndPower
# parameters for power analysis
effect = 0.7
alpha = 0.05
power = 0.8
nobs1=26
# perform power analysis
analysis = TTestIndPower()
# this function requires 4 variables, all of which are related. One of them must be 'none' in order for it to generate an answer
result = analysis.solve_power(effect, power=None, nobs1=nobs1, ratio=1.0, alpha=alpha)
print(result)

################################################################################################################################################    