#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:57:33 2020

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
from time import time

group_names=['Session','Singer','Reg+gen','Register','Gender']
#variables group is itself one of the condition groups that contain variables
for group_idx, variables_group in enumerate(averaged_groups):
    varialble_specific_conditions_group = []
    for variable_idx, conditions_group in enumerate(variables_group):
        # looking at all the values for this group
        condition_specific_value_avgs_group = []
        for values_group in conditions_group:
            value_sum = 0
            for val_idx, value in enumerate(values_group):
                value_sum += value
            value_avg = value_sum/(val_idx+1)
            condition_specific_value_avgs_group.append(value_avg)
        varialble_specific_conditions_group.append(condition_specific_value_avgs_group)
    
    varialble_specific_conditions_group_array = np.asarray(varialble_specific_conditions_group)
    male_class_dissim_avgs = varialble_specific_conditions_group_array[:,0]
    female_class_dissim_avgs = varialble_specific_conditions_group_array[:,1]
    # fill class*class dissim matrix with averages
        
    
    # how many columns are in the array? Thats how many conditions there are
    for condition_int in range(varialble_specific_conditions_group_array.shape[1]):
        class_dissim_matrix = np.empty((5,5))
        conditioned_variables_array = varialble_specific_conditions_group_array[:,condition_int]
        # this will always be the correct order for every class-distance matrix
        class_dissim_matrix[0,1] = conditioned_variables_array[0]
        class_dissim_matrix[0,2] = conditioned_variables_array[1]
        class_dissim_matrix[0,3] = conditioned_variables_array[2]
        class_dissim_matrix[0,4] = conditioned_variables_array[3]
        class_dissim_matrix[1,2] = conditioned_variables_array[4]
        class_dissim_matrix[1,3] = conditioned_variables_array[5]
        class_dissim_matrix[1,4] = conditioned_variables_array[6]
        class_dissim_matrix[2,3] = conditioned_variables_array[7]
        class_dissim_matrix[2,4] = conditioned_variables_array[8]
        class_dissim_matrix[3,4] = conditioned_variables_array[9]
        class_dissim_matrix[0,0] = conditioned_variables_array[10]
        class_dissim_matrix[1,1] = conditioned_variables_array[11]    
        class_dissim_matrix[2,2] = conditioned_variables_array[12]
        class_dissim_matrix[3,3] = conditioned_variables_array[13]
        class_dissim_matrix[4,4] = conditioned_variables_array[14]
        
        for i in range(5):
            for j in range(5):
                class_dissim_matrix[j,i] = class_dissim_matrix[i,j]
        if group_idx==4:
            if condition_int==0:
                title = 'Male'
            else:
                title = 'Female'
        elif group_idx==3:
            if condition_int==0:
                title = 'Low Register'
            else:
                title = 'High Register'
        else:
            title = 'Condition Group ' +str(group_idx) +', Condition ' +str(condition_int)
        util.mds_from_dissim_matrix(class_dissim_matrix, label_list, 2, show_plot=True, title=title)
#        util.mds_from_dissim_matrix(class_dissim_matrix, label_list, 3, show_plot=True, title=title)