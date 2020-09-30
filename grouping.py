#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 20:54:03 2020

@author: brendanoconnor
"""

import os, h5py, util
from xml.etree import ElementTree
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


if 'h5' in globals():
    h5.close()

h5 = h5py.File('anonomisedData.hdf5', mode = 'r')

##########################################################################################

num_experiments = len(h5['participantRatings'])
total_features = 8
profile_array = np.empty((0,total_features))
for i in range(num_experiments):
    p = h5['participantInfo'][i]
    row = np.array([i,     #index
                    p[20], #hearImpairments
                    p[22], #listening_env
                    p[24], #musician_category
                    p[23], #msi_score
                    p[25], #task comprehension
                    p[26], #similarity identification (zero_error)
                    p[27]])#consistency
    row = row.reshape(-1, total_features)
    profile_array = np.append(profile_array, row, axis=0)

# next is to see if theres correlation between any columns in profile_array
    
num_columns = len(profile_array[0])
profile_array = profile_array.astype(np.float)

# GET SLICE OF NUMPY, SORT THE INDICES OF THAT NUMERICALLY, APPLY THESE NEWLY SORTED INDICES TO THE ORIGINAL NUMPY, YOU GET REDORDERED NUMPY
reordered_profile_array = profile_array[(-profile_array)[:,6].argsort()]



#for profile_scores in reordered_profile_array:
#    if profile_scores[7]>0.347:
#        tester.append(profile_scores[0])
#    if profile_scores[5]==0:
#        poor_scores_indices.append(profile_scores[0])
#    if profile_scores[5]==1:
#        poor_scores_indices.append(profile_scores[0])
        
##########################################################################################
# REMOVE THOSE THAT SCORERS
poor_scores_indices = []
good_profile_array = profile_array.copy()
for row_idx, row in enumerate(profile_array):
    if row[0] in poor_scores_indices:
        good_profile_array = np.delete(good_profile_array, row_idx, axis=0)
            
    
##########################################################################################
# SORT PROFILE ARRAY BY COLUMN
# hand-picked participants that produced erractic results

bad_profile_array = np.empty((0, 8))
for i in range(len(profile_array)):
    if i in bad_boxing:
        reshaped_row = profile_array[i].reshape(-1,8)
        bad_profile_array = np.append(bad_profile_array, reshaped_row, axis=0)
bad_profile_array = bad_profile_array[(-bad_profile_array)[:,7].argsort()] 
    
##########################################################################################
## PARTICIPANT PROFILE FILTERING
#
## REMOVE ANYONE WITH LISTENING ENVIRONMENT SCORE BELOW X
#filtered_profile_array=filtered_profile_array[profile_array[:,3]>5]
#
## REMOVE ANYONE WITH CONSISTENCY BELOW X
#filtered_profile_array=filtered_profile_array[profile_array[:,7]>0.3]
#
## REMOVE ANYONE WITH MSI SCORE BELOW X
#filtered_profile_array=filtered_profile_array[profile_array[:,4]>62.0]
#
## REMOVE ANYONE WITH TASK COMPREHENSION BELOW X (scale 0 - 5)
#filtered_profile_array=filtered_profile_array[profile_array[:,4]>1]
## REMOVE ANYONE WITH UNKNOWN TASK COMPREHENSION CATEGORY '5'
#filtered_profile_array=filtered_profile_array[profile_array[:,4]==5]

###########################################################################################
## MUSICIAN FILTERING
#
## REMOVE SINGERS
#filtered_profile_array=filtered_profile_array[profile_array[:,3]!=2]
#
## REMOVE NON-SINGER MUSICIANS
#filtered_profile_array=filtered_profile_array[profile_array[:,3]!=1]
#
## REMOVE NON-MUSICIANS
#filtered_profile_array=filtered_profile_array[profile_array[:,3]!=0]

###########################################################################################
### SESSION FILTERING

indices_for_low_sessions = util.indices_by_key_value(h5, 'participantInfo', 1, 'low')
indices_for_high_sessions = util.indices_by_key_value(h5, 'participantInfo', 1, 'high')
indices_for_male_sessions = util.indices_by_key_value(h5, 'participantInfo', 1, 'm')
indices_for_female_sessions = util.indices_by_key_value(h5, 'participantInfo', 1, 'f')

indices_for_low_sessions = np.setdiff1d(indices_for_low_sessions, poor_scores_indices)
indices_for_high_sessions = np.setdiff1d(indices_for_high_sessions, poor_scores_indices)
indices_for_male_sessions = np.setdiff1d(indices_for_male_sessions, poor_scores_indices)
indices_for_female_sessions = np.setdiff1d(indices_for_female_sessions, poor_scores_indices)

indices_for_low_male_session = np.setdiff1d(indices_for_male_sessions, indices_for_high_sessions)
indices_for_high_male_session = np.setdiff1d(indices_for_male_sessions, indices_for_low_sessions)
indices_for_low_female_session = np.setdiff1d(indices_for_female_sessions, indices_for_high_sessions)
indices_for_high_female_session = np.setdiff1d(indices_for_female_sessions, indices_for_low_sessions)

sessions_indices_lists = []
sessions = ['m1low','m2low','m4low','m1high','m2high','m4high','f2low','f3low','f5low','f2high','f3high','f5high']
for session in sessions:
    indices_for_specific_session = util.indices_by_key_value(h5, 'participantInfo', 1, session)
    indices_for_specific_session_no_bads = np.setdiff1d(indices_for_specific_session, poor_scores_indices).tolist()
    sessions_indices_lists.append(indices_for_specific_session_no_bads)

singers_indices_lists = []
singers = ['m1','m2','m4','f2','f3','f5']
for singer in singers:
    indices_for_specific_singer = util.indices_by_key_value(h5, 'participantInfo', 1, singer)
    indices_for_specific_singer = np.setdiff1d(indices_for_specific_singer, poor_scores_indices).tolist()
    singers_indices_lists.append(indices_for_specific_singer)
    
registers_indices_lists = [indices_for_low_sessions, indices_for_high_sessions]
genders_indices_lists = [indices_for_male_sessions, indices_for_female_sessions]

subgroups_indices_list = [sessions_indices_lists, singers_indices_lists, registers_indices_lists, genders_indices_lists]
