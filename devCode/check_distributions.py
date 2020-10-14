#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 14:59:21 2020

@author: brendanoconnor

"""

import util, h5py, os
import numpy as np
from time import time
import matplotlib
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
import math, itertools
import scipy.stats as stats

if 'h5' in globals():
    h5.close()
h5 = h5py.File('/Users/brendanoconnor/Desktop/studyData20200707.hdf5', mode = 'r')
num_experiments = len(h5['pageInfo'])
show_save = True
label_list = ['straight','belt','breathy','fry','vibrato']
sessions = ['m1low','m2low','m4low','m1high','m2high','m4high','f2low','f3low','f5low','f2high','f3high','f5high']
singers = ['m1','m2','m4','f2','f3','f5']
registers = ['lows','highs']
genders = ['male','female']
class_list = [0,1,2,3,4]
class_pairs = list(itertools.combinations(class_list, 2))
class_pairs.extend(((0,0), (1,1), (2,2), (3,3), (4,4)))

# PLOT DISTRIBUTIONS FOR ALL CLASS COMPARISONS FOR EACH PARTICIPANT, AND COLLECT RATINGS FROM GROUP INDICES
intra_session_ratings = util.class_rating_distributions(h5, sessions_indices_lists, class_pairs, sessions, 'intra_session_distributions', show_save)        
intra_singer_ratings = util.class_rating_distributions(h5, singers_indices_lists, class_pairs, singers, 'intra_singer_distributions', show_save)
intra_register_ratings = util.class_rating_distributions(h5, registers_indices_lists, class_pairs, registers, 'intra_register_distributions', show_save)
intra_gender_ratings = util.class_rating_distributions(h5, genders_indices_lists, class_pairs, genders, 'intra_gender_distributions', show_save)

# PLOT DISTRIBUTION OF RATINGS PER PARTICIPANT - SHOWS HOW FREQUENTLY THEY USED CERTAIN RATINGS
square_matrix_dim = len(h5['dissimMatrix'][0])
flattened_participant_ratings_list = []
directory_name = 'participant_rating_distributions'
for experiment_idx in range(num_experiments):
    dissim_matrix = h5['rearrangedDissimMatrix'][experiment_idx]
    flattened_dissim_matrix = dissim_matrix.flatten()
    flattened_participant_ratings_list.append(flattened_dissim_matrix)
    if show_save == True:
        plt.hist(flattened_dissim_matrix, bins=15)
        os.makedirs(directory_name, exist_ok=True)
        title ='Experiment_idx {0} ratings'.format(experiment_idx)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(directory_name +'/' +'Experiment_idx {0} ratings'.format(experiment_idx))
        plt.close()
        
# PLOT DISTRIBUTION FOR RATINGS ACROSS ALL PARTICIPANTS
flattened_flattened_participant_ratings_list = [rating for participant in flattened_participant_ratings_list for rating in participant]
if show_save == True:
    plt.hist(flattened_flattened_participant_ratings_list, bins=15)
    os.makedirs(directory_name, exist_ok=True)
    title ='All Participant Ratings'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(directory_name +'/' +'All Participant Ratings')
    plt.show()
    plt.close()
    
# PLOT DISTRIBUTION OF VALUES FOR EACH PROFILE FEATURE ACROSS ALL PARTICIPANTS
# Each column in the profile array represents participant profile measurement. 

directory_name = 'participant_profile_distributions'
for profile_feature in range(1,8):
    profile_feature_list = []
    for experiment_idx in range(num_experiments):
        profile_feature_list.append(profile_array[experiment_idx][profile_feature])
    if show_save == True:
        plt.hist(profile_feature_list, bins=10)
        os.makedirs(directory_name, exist_ok=True)
        title ='Profile Array column {0} distribution'.format(profile_feature)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(directory_name +'/' +'Profile Array column {0} distribution'.format(profile_feature))
        plt.show()
        plt.close()    

# PLOT DISTRIBUTION OF AGES ACROSS ALL PARTICIPANTS
ages = []
male_ages=[]
female_ages=[]
for experiment_idx in range(num_experiments):
    ages.append(int(h5['participantInfo'][experiment_idx][7]))
    if h5['participantInfo'][experiment_idx][8]=='m':
        gender = 0
        male_ages.append(int(h5['participantInfo'][experiment_idx][7]))
    elif h5['participantInfo'][experiment_idx][8]=='f':
        gender = 1
        female_ages.append(int(h5['participantInfo'][experiment_idx][7]))
    else:
        gender = 'error'
    genders.append(gender)
if show_save == True:
    plt.hist([male_ages, female_ages], stacked=True, bins=10)
    os.makedirs(directory_name, exist_ok=True)
    title ='Age distribution'
    plt.title(title)
    plt.tight_layout()
    plt.xticks(range(20, 75, 5))
    labels = ['male', 'female']
    plt.legend(labels)
    plt.savefig(directory_name +'/' +'Distribution of Ages')
    plt.show()
    plt.close()
male_ages_array = np.asarray(male_ages)
female_ages_array = np.asarray(female_ages)
all_ages = np.concatenate((male_ages_array, female_ages_array))
print('Participant age mean = {0}, std = {1}.'.format(np.mean(all_ages), np.std(all_ages)))

# PLOT GRAPH OF THE THREE CATEGORIES OF INSTRUMENTAL ABILITY USED IN THIS STUDY
men_non_mus = 0
men_mus = 0
men_singer_mus= 0
women_non_mus = 0
women_mus = 0
women_singer_mus= 0
N = 3
ind = np.arange(N)
for experiment_idx in range(num_experiments):
    if h5['participantInfo'][experiment_idx][24]=='0':
        if h5['participantInfo'][experiment_idx][8]=='m':
            men_non_mus += 1
        else:
            women_non_mus +=1
    elif h5['participantInfo'][experiment_idx][24]=='1':
        if h5['participantInfo'][experiment_idx][8]=='m':
            men_mus += 1
        else:
            women_mus +=1
    elif h5['participantInfo'][experiment_idx][24]=='2':
        if h5['participantInfo'][experiment_idx][8]=='m':
            men_singer_mus += 1
        else:
            women_singer_mus +=1
if show_save == True:
    p1 = plt.bar(ind, (men_non_mus, men_mus, men_singer_mus))
    p2 = plt.bar(ind, (women_non_mus, women_mus, women_singer_mus), bottom=(men_non_mus, men_mus, men_singer_mus))
    os.makedirs(directory_name, exist_ok=True)
    title ='Instrumental Ability'
    plt.title(title)
    plt.xticks(ind, ('Non-Mus', 'Mus', 'Singers'))
    plt.tight_layout()
    labels = ['male', 'female']
    plt.legend((p1[0], p2[0]), ('male', 'female'))
    plt.savefig(directory_name +'/' +'Instrumental Ability')
    plt.show()
    plt.close()

# PLOT GENDER GRAPH
male = 0
female = 0
for experiment_idx in range(num_experiments):
    if h5['participantInfo'][experiment_idx][8]=='m':
        male += 1
    elif h5['participantInfo'][experiment_idx][8]=='f':
        female += 1
genders = [male, female]
labels = ['male', 'female']
if show_save == True:
    plt.bar(labels, genders)
    os.makedirs(directory_name, exist_ok=True)
    title ='Participant Genders'
    plt.title(title)
    plt.tight_layout()
    plt.savefig(directory_name +'/' +'Participant Genders')
    plt.show()
    plt.close()  


##################################################################################################################################
# CHECK FOR NORMAL DISTRIBUTION USING SHAPIRO WILKS AND KLOMOGOROV-SMIRNOV TESTS
rating_groups = [intra_session_ratings,
                 intra_singer_ratings,
                 intra_register_ratings,
                 intra_gender_ratings]

rating_groups_skewness = []
rating_groups_ws = []
rating_groups_ks = []

ws_p_05_normals = []
ws_p_10_normals = []
for group_idx, grouping in enumerate(rating_groups):
    data_skewness = []
    data_ws = []
    data_ks = []
    for class_int, class_comparison_ratings in enumerate(grouping):
        for condition_int, condition in enumerate(class_comparison_ratings):
            flattened_list = [rating for participant in condition for rating in participant]
            skewness = stats.skew(flattened_list)
            # SHAPIRO-WILK - if it returns a p value less than the accepted 0.05, data is NOT normal
            W, p = stats.shapiro(flattened_list)
            if p>=0.10:
                ws_p_10_normals.append(('group,class,condition,W,p ', group_idx, class_pairs[class_int], condition_int, round(W, 5), round(p, 5)))
            if p>=0.05:
                ws_p_05_normals.append(('group,class,condition,W,p ', group_idx, class_pairs[class_int], condition_int, round(W, 5), round(p, 5)))
    #        print('Shapiro Wilks: W{0} p={1}'.format(W,p))
            D, p2 = stats.kstest(flattened_list, 'norm')
    #        print('Klomogorov-Smirnov: W{0} p={1}'.format(D,p2))
            data_skewness.append((class_pairs[class_int], condition_int, skewness))
            data_ws.append((class_pairs[class_int], condition_int, W, p))
            data_ks.append((class_pairs[class_int], condition_int, D, p2))
    rating_groups_skewness.append(data_skewness)
    rating_groups_ws.append(data_ws)
    rating_groups_ks.append(data_ks)

##################################################################################################################################
# GENERATE AVERAGE DISTANCES FOR CLASS PAIRS

directory_name = 'average_class_distance_distributions'
averaged_groups = []
for rating_group_idx, rating_group in enumerate(rating_groups):
    conditions_list = []
    for class_group_idx, class_group in enumerate(rating_group):
        # get the average of all the distances for each condition of each class of each group  
        participant_averages_list=[]
        for condition_group_idx, condition_group in enumerate(class_group):
            # convert multiple scores for each participant into a single average for them
            participant_averages = []
            for part_idx, participant in enumerate(condition_group):
                num_ratings = len(participant)
                summed_results=0.0
                for rating in participant:
                    summed_results+=rating
                average = summed_results/num_ratings
                participant_averages.append(average)
            # add this condition to the group for this specific class_comparison
            if show_save == True:
                plt.hist(participant_averages, bins=10)
                os.makedirs(directory_name, exist_ok=True)
                title ='Group {0}, Condition {2}, Class pair {1}, Averages Distribution'.format(rating_group_idx, class_pairs[class_group_idx], condition_group_idx)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(directory_name +'/' +title)
                plt.show()
                plt.close()   
            participant_averages_list.append(participant_averages)
        # add this class to the group for this rating groups
        conditions_list.append(participant_averages_list)
    averaged_groups.append(conditions_list)
