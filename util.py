#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:38:02 2020

@author: brendanoconnor
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import load_digits
from sklearn.manifold import MDS
import h5py, math, os, itertools
import numpy as np
import scipy.stats as stats
from statistics import mean, variance
from scipy.stats import pearsonr, kendalltau, spearmanr, ttest_ind, rankdata, chi2
from kneed import KneeLocator

############################################################################################################
# GET A LIST OF LISTS OF INDICES AND GO THROUGH ALL CLASS-COMBINATION RATINGS
# GET DISTRIBUTION FOR EVERY CLASS COMBO FOR EACH CONDITIONS SEPARATELY
# PLOT AND SAVE HISTOGRAM, RETURN RATINGS IN LIST OF LISTS FORMAT

def class_rating_distributions(h5, parent_list, class_pairs, id_string_list, directory_name, show_plot=True):
    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    all_class_distances_all_lists = []
    for ref_class, comp_class in class_pairs:
        specific_class_all_lists = []
        for parent_idx, sublist in enumerate(parent_list):
            sublist_distances = []
            for experiment_idx in sublist:
#                print('idx: ', idx, 'participantInfo: ', h5['participantInfo'][idx])
                rearranged_dissim_matrix = h5['rearrangedDissimMatrix'][experiment_idx]
                rearranged_ref_audio = h5['rearrangedReferenceAudioNames'][experiment_idx]
                rearranged_class_list = strings_to_classes(rearranged_ref_audio, label_list)

                specific_participant_distances = []
                for class_idx_row, class_val_row in enumerate(rearranged_class_list):
                    if class_val_row == ref_class:
                        matrix_row_val = class_idx_row
                        for class_idx_col, class_val_col in enumerate(rearranged_class_list):
                            if class_val_col == comp_class:
                                matrix_col_val = class_idx_col
        #                        print('class_idx_row', class_idx_row, 'class_idx_col', class_idx_col)
                                comparison = rearranged_dissim_matrix[matrix_row_val,matrix_col_val]
                                specific_participant_distances.append(comparison)
                sublist_distances.append(specific_participant_distances)
                os.makedirs(directory_name +'/' +id_string_list[parent_idx], exist_ok=True)
            # flatten list of lists to long list - see https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
            specific_class_all_lists.append(sublist_distances)
            flattened_distances = [rating for sublist in sublist_distances for rating in sublist]
            # PLOT AND SAVE HISTOGRAM
            if show_plot==True:
                plt.hist(flattened_distances, bins=10)
                title ='class {0} to {1} ratings'.format(ref_class, comp_class)
                plt.title(title)
                plt.tight_layout()
                plt.savefig(directory_name +'/' +id_string_list[parent_idx] +'/' +title)
                plt.show()
                plt.close()
        all_class_distances_all_lists.append(specific_class_all_lists)
    all_class_distances_all_lists = np.asarray(all_class_distances_all_lists)
    np.save(directory_name +'/' +directory_name, all_class_distances_all_lists)
    return all_class_distances_all_lists

############################################################################################################

def kmeans_sse_from_arrays(k_range, arrays):
    sum_squared_error = []
    for k_dim in k_range:
        km = KMeans(n_clusters=k_dim)
        km.fit_predict(arrays)
        sum_squared_error.append(km.inertia_)
    return sum_squared_error

############################################################################################################
# the closer the decreasing factor is to 1, the more minimal a decrease in the cluster diffs is allowed to be
# this returns the number of dvalue of dimensions found at the elbow.

def find_elbow_dimension(cluster_scores, decreasing_factor):
    # put diffs between cluster scores in list
    cluster_diffs=[]
    for idx, clusterscore in enumerate(cluster_scores):    
        if idx==0:
            continue
        cluster_diffs.append(cluster_scores[idx-1]-clusterscore)
    
    elbow_diff_idx=100
    elbow_dim_idx=-1
    for idx, diff in enumerate(cluster_diffs):
#        print('{0} less than {1}/{2}...'.format(diff, cluster_diffs[idx-1], decreasing_factor))
        if diff<(cluster_diffs[idx+1]*decreasing_factor):
            continue
        else:
            # because the 0th idx corresponds to 1dim, and on the 0th index we are seeing if there is an elbow between 1 and 2
            # if there is,then the 0th idx would correspond to dimension 2. Therefore +2 is required
            elbow_dim = idx+2
            break
    return elbow_dim

################################################################################################
# ANALYSE CLUSTERING SCORE

def cluster_scoring(idx, dissim_matrix, k_range, show_plot=True):
    sse_list=[]
    sse = kmeans_sse_from_arrays(k_range, dissim_matrix)
    if show_plot==True:
        fig_1 = plt.figure(1, figsize=(6.4, 4.8))
        plt.plot(k_range, sse)
#        plt.title('SSE per k number for Exp ' +str(idx))
        plt.xlabel('Dimensions for k')
        plt.ylabel('SSE loss')
        plt.savefig('kMeans_SSEs_default_setting/' +str(idx))
        plt.show()
        plt.close()
    return sse

################################################################################################
# ASSIGN CLUSTERS
    
def assign_clusters(dissim_matrix, audio_refs, k):
#    num_audio = len(dissim_matrix)
    km = KMeans(n_clusters=k)
    cluster_prediction = km.fit_predict(dissim_matrix)    
    cluster_prediction = cluster_prediction.reshape(len(cluster_prediction),-1)
    audio_refs = audio_refs.reshape(len(audio_refs),-1)
    clustered_audio_refs = np.append(audio_refs, cluster_prediction, axis=1)
    return clustered_audio_refs

################################################################################################
    
def get_elbow_array(h5, indices, k_range, file_name, show_plots, s_param):
    num_experiments = len(indices)
    elbows=np.empty(num_experiments)
    for idx_idx, idx in enumerate(indices):
        rearranged_dissim_matrix = h5['rearrangedDissimMatrix'][idx]
        dissim_matrix = np.around(rearranged_dissim_matrix, 2)    
        cluster_scores = cluster_scoring(idx, dissim_matrix, k_range, show_plot=show_plots)
        kneedle = KneeLocator(k_range, cluster_scores, S=s_param, curve='convex', direction='decreasing')
    #    print('idx: ', idx,'elbow_dim: ', kneedle.elbow)
        elbows[idx_idx] = kneedle.elbow
    if show_plots==True:
        elbow_bar_solutions = []
        # get range between (inclusive means adding +1) min and max
        cluster_range=range(int(np.min(elbows)),int(np.max(elbows)+1))
        for entry in cluster_range:
            cluster_num = entry
            total_sum_per_cluster_num = np.count_nonzero(elbows==cluster_num)
            elbow_bar_solutions.append(total_sum_per_cluster_num)
            
        plt.bar(cluster_range, elbow_bar_solutions)
        plt.title(file_name)
        plt.xticks(k_range)
#        plt.yticks(range(0,40,2))
        plt.tight_layout()
        plt.savefig('kMeans_elbow_S0p25_distributions/' +file_name)
        plt.show()
        plt.close()
#        print('With S= ' +str(s_level))
    return elbows

################################################################################################

def get_k_array_from_silhouettes (h5, indices, k_range, file_name, show_plots):
    num_experiments = len(indices)
    best_k_from_silhouettes = np.empty(num_experiments)
    os.makedirs('k_silhouettes_distributions/', exist_ok=True)
    for idx_idx, idx in enumerate(indices):
        silhouettes = np.empty(len(k_range))
        rearranged_dissim_matrix = h5['rearrangedDissimMatrix'][idx] 
        for k_idx, num_clusters in enumerate(k_range):
            hc = AgglomerativeClustering(n_clusters=num_clusters, affinity = 'euclidean', linkage='ward')
            hc_clustering = hc.fit_predict(rearranged_dissim_matrix)
            hc_sil_score = silhouette_score(rearranged_dissim_matrix, hc_clustering, metric='euclidean')
    #            print('sil value', num_clusters, hc_sil_score)
            silhouettes[k_idx] = hc_sil_score
    #    if True==True:
    #        plt.plot(silhouettes)
    #        plt.title('file_name' +str(idx))
    #        plt.show()
    #        plt.close()
    
        # add 2 because the k_range 0th index is 2
        best_k_from_silhouettes[idx_idx] = np.argmax(silhouettes)+2
    if show_plots==True:
        # turn best_k_from_silhouettes into bar graph suitable data
        sil_bar_solutions = []
        cluster_solution_range = range(int(np.min(best_k_from_silhouettes)),int(np.max(best_k_from_silhouettes)+1))
        for entry in cluster_solution_range:
            cluster_num = entry
            total_sum_per_cluster_num = np.count_nonzero(best_k_from_silhouettes==cluster_num)
            sil_bar_solutions.append(total_sum_per_cluster_num)
            
        plt.bar(cluster_solution_range, sil_bar_solutions)
        plt.title(file_name)
        plt.xticks(k_range)
        plt.xlabel('Dimensions for k')
        # check if this is enough
#        plt.yticks(range(0,40,2))
        plt.tight_layout()
        plt.savefig('k_silhouettes_distributions/' +file_name)
        plt.show()
        plt.close()
    return best_k_from_silhouettes

################################################################################################

def array_from_sesh_name(h5, num_experiments, sesh_name):
    # collect all sessions with target session name
    session_list=[]
    square_matrix_dim = len(h5['dissimMatrix'][0])
    for experiment_idx in range(num_experiments):
        if sesh_name in h5['participantInfo'][experiment_idx][1]:
            session_numpy = h5['dissimMatrix'][experiment_idx]
            session_numpy = np.reshape(session_numpy, (-1, square_matrix_dim*square_matrix_dim))
            session_numpy = np.squeeze(session_numpy)
            session_list.append(session_numpy)
    sessions_array = np.asarray(session_list)
    return sessions_array

################################################################################################

# replace above function with this when tested and confirmed it works the same
def array_by_h5_value(h5, key, index, value):
    # collect all sessions with target session name
    session_list=[]
    num_experiments = len(h5[key])
    square_matrix_dim = len(h5['dissimMatrix'][0])
    for experiment_idx in range(num_experiments):
        if h5['participantInfo'][experiment_idx][index] == value:
            session_numpy = h5['dissimMatrix'][experiment_idx]
            session_numpy = np.reshape(session_numpy, (-1, square_matrix_dim*square_matrix_dim))
            session_numpy = np.squeeze(session_numpy)
            session_list.append(session_numpy)
    sessions_array = np.asarray(session_list)
    # this isn't a matrix - fix this (its a 1*196 list in a list)
    return sessions_array

############################################################################################################

def generate_dissim_matrix(h5, experiment_idx):
    # fake values to see which cells are not filled in with real ratings
    dummy_value = 100.01
    num_audio = len(h5['participantRatings'][0])
    dissimilarity_matrix = np.full((num_audio, num_audio), dummy_value)
    # make sure to always process pages by order of the page number (not the presentedId number)
    # page_num will be our row for the dissim matrix too and the index for the refAudio
    # we are adding 1 on here because pageRef '13' is missing but we need the page_num to match pageRef '14' as well
    for page_num in range(num_audio+1):
        
        # page_idx tells us which row to address for other h5 groups
        for page_idx, pageInfo in enumerate(h5['pageInfo'][experiment_idx]):

            if page_num==int(pageInfo[2]):
#                print('pageRef', pageInfo[2])
                # remember, page_idx is the ref_idx toooooo!
                dissim_row = page_idx
#                print('page_idx', page_idx)
                # whatever the page is, the same referenceAudio will always be linked to it
                # so the order in which we process referenceAudio will still be the same
                refAudioName = h5['referenceAudioNames'][experiment_idx][page_idx]
                # is the unwanted duplicate data here as a ref audio?
#                if refAudioName == duplicate_audio_name:
#                    continue
    #            print('refAudioName',refAudioName)
                page_ratings = h5['participantRatings'][experiment_idx][page_idx]
                for rating_idx, rating in enumerate(page_ratings):
                    compAudioName = h5['comparativeAudioNames'][experiment_idx][page_idx][rating_idx]
                    # if the comp audio is the neglected audio name, ignore it
                    if compAudioName == 'neglected_audio_name':
#                        print('neglected_audio_name found')
                        # do nothing and go to next rating
                        continue
                    
                    for ref_idx, refName in enumerate(h5['referenceAudioNames'][experiment_idx]):
                        if refName==compAudioName:
                            dissim_column = ref_idx
#                            print(page_idx,ref_idx)
#                            print('dissim_row, dissim_column, refAudio, compAudio, rating', (dissim_row, dissim_column), (refAudioName, compAudioName), rating)
                            dissimilarity_matrix[dissim_row, dissim_column]=rating.round(decimals=2)
                            break
                break
                    
    # turn half-filled dissimilarity matrix into full disimilarity matrix
    for row in range(num_audio):
        for col in range(num_audio):
            if dissimilarity_matrix[row][col]==dummy_value:
                dissimilarity_matrix[row][col] = dissimilarity_matrix[col][row]
                
    return dissimilarity_matrix

############################################################################################################
# GO THROUGH ARRAY AND FIND OUT ORDER OF MATRICES - IE WHICH AUDIO FILES CORRESPOND TO WHICH COLUMN
# RETURNS A LIST OF [indices, LABELS]
# first entry in dissimilarity matrix array is the reference, the rest are rearranged to it
# idx list is a list of participant indices in the hdf5 file that we want for referencing different hdf5 attributes
def reordered_list_between_sessions(h5, dismat_array, idx_list):
    
    label_dict = {'straight':0, 'belt':1, 'breathy':2, 'fry':3, 'vibrato':4}
#    color_list = ['r', 'g', 'b', 'c', 'm']
    
    # use first entry from list as the matrix to check others against WARNING: these h5 sections are double-listed ;)
    ref_ref_list = h5['referenceAudioNames'][idx_list[0]][0]
    print('ref_ref_list',ref_ref_list)
    # element at index informs which ref
    idx_for_ref_lists = []
    # for each matrix in disim_array
    for dismat_idx, dismat in enumerate(dismat_array):
        # we are using the 0th entry as the reference and sync up the other matrices to it (of course the 0th matrix will be linked to itself)
        comp_ref_list = h5['referenceAudioNames'][idx_list[dismat_idx]][0]
        matrix_refs=[]
        # for each ref in the pre-loop-established ref list
        for ref_ref_idx, ref_ref_name in enumerate(ref_ref_list):
            # for each name in the comp list
            for comp_ref_idx, comp_ref_name in enumerate(comp_ref_list):
                # if names same
                if ref_ref_name==comp_ref_name:
                    # log the idx of the comp list and move onto next ref
                    for label in label_dict:
                        if label in comp_ref_name:
                            matrix_refs.append([comp_ref_idx,label_dict[label]])
#                            print(comp_ref_name, comp_ref_idx,label_dict[label])    
                    break
        # save each of these to list of lists
        # WARNING: THE ELEMENTS INDEX LISTS TELL US WHICH INDEX from the comp-ref-list
        # (and therefore which matrix row) matches with that of the ref-ref-list
        idx_for_ref_lists.append(matrix_refs)
    return idx_for_ref_lists


############################################################################################################
# GO THROUGH ARRAY AND FIND OUT ORDER OF VECTOR - IE WHICH AUDIO FILES CORRESPOND TO WHICH COLUMN
# matches the comp array against the ref array, and provides sequence of reference indices
def get_reordered_index_for_repeat(ref_array, comp_array):
    idx_for_ref_list = []
    for repeat_idx, repeat_comp_audio in enumerate(comp_array):
        for orginal_idx, original_comp_audio in enumerate(ref_array):
            if repeat_comp_audio==original_comp_audio:
                idx_for_repeat = orginal_idx
                idx_for_ref_list.append(idx_for_repeat)
                break
    return idx_for_ref_list


############################################################################################################


def get_rmse_between_repeats(original_ratings, repeat_ratings, idx_for_ref):
        # make empty numpy of dimensions num_exp, no of repeat pages, number of difference measurements
#        reliability_scores = np.empty((num_experiments, 2, 3))
#        reliability_scores = []
        # collect differences in a list
    squared_diff_sum = 0
    for idx in range(len(original_ratings)):
        original_rating = original_ratings[idx_for_ref[idx]]
        repeat_rating = repeat_ratings[idx]
#        print('original rating: ',original_rating, 'repeat rating: ', repeat_rating)
        squared_diff = (original_rating-repeat_rating)*(original_rating-repeat_rating)
#        print('squared diff: ', squared_diff)
        squared_diff_sum += squared_diff
#                abs_diff = abs(original_rating-repeat_rating)
#                abs_diff_sum += abs_diff
#            mae = abs_diff_sum/len(original_ratings[ratings_idx])
#    print('squared_diff_sum: ',squared_diff_sum)
    mse = squared_diff_sum/len(original_ratings)
    rmse = math.sqrt(mse)
#    print('rmse',rmse )
#        reliability_scores_list.append(reliability_scores)
    return rmse


################################################################################################################################
# FOR PLOTTING AND RETURN MDS VALUES


def mds_from_dissim_matrix(dissim_matrix, ref_list, dimensions, show_plot=True, title='mds'):
    np.random.seed(19680801)
#    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    label_dict = {'straight':0, 'belt':1, 'breathy':2, 'fry':3, 'vibrato':4}
    color_list = ['r', 'g', 'b', 'c', 'm']

#    print(dissim_matrix.shape)
    embedding = MDS(n_components=dimensions, metric=False, dissimilarity='precomputed')
    exp_transfomed = embedding.fit_transform(dissim_matrix)
    os.makedirs('mds_plots/', exist_ok=True)
#    print(exp_transfomed.shape)
    if show_plot==True:
        
        fig = plt.figure()
        if dimensions==3:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)
        # match labels to ref_list to assign colours to scatter plot points
        for label_int, label in enumerate(label_dict):
            for ref_ind, referenceAudioName in enumerate(ref_list):  
                if label in referenceAudioName:
                    label_colour = color_list[label_int]
                    if dimensions==3:
                        ax.scatter(exp_transfomed[ref_ind, 0], exp_transfomed[ref_ind, 1], exp_transfomed[ref_ind, 2], c=label_colour, label=label)
                    else:
                        ax.scatter(exp_transfomed[ref_ind, 0], exp_transfomed[ref_ind, 1], c=label_colour, label=label)
                        
        plt.xlim(-0.8,0.8)
        plt.ylim((-0.7,0.7))
        plt.title(title)
        plt.legend(label_dict)
        plt.savefig('mds_plots/' +title)
        plt.show()
        plt.close()
    return exp_transfomed

############################################################################################################
# MAKE MATRIX WITH ZERO DIAGONAL AND RETURN ERROR

def zeroed_dissim_matrix(dissim_matrix, num_audio):
    sum_error = 0.
    for index in range(num_audio):
        if dissim_matrix[index][index]!=0.:
            sum_error += dissim_matrix[index][index]
            dissim_matrix[index][index]=0.
    return dissim_matrix, sum_error


############################################################################################################
# MAKE NEW REORDERED MATRICES USING THE INFO LEARNED ABOVE

#def reorder_matrix_array(h5, dismat_array, idx_for_ref_lists):
## make placeholder variable for new matrices and add the key matrix first
#    square_matrix_dim = len(h5['dissimMatrix'][0])
#    rearranged_matrices = np.empty((0,square_matrix_dim,square_matrix_dim), float)
#    first_matrix = np.reshape(dismat_array[0], (-1, square_matrix_dim, square_matrix_dim))
#    rearranged_matrices = np.append(rearranged_matrices, first_matrix, axis=0)
#    
#    # for each matrix in matrix array
#    for mat_idx in range(1, len(dismat_array)):
#        current_matrix = dismat_array[mat_idx]
#        rearranged_matrix = np.empty((0,square_matrix_dim), float)
#        # get cocrresponding idx list
#        target_indices = idx_for_ref_lists[mat_idx]  
#        
#        #go through each row and take the ccorresponding row from the key matrix
#        # BUT THIS IS NOT ENOUGH - MUST ALSO REARRANGE ROW TO THE SAME INDEX VECTOR AS THE COLUMNS
#        for row_idx in range(square_matrix_dim):
#            target_index = target_indices[row_idx][0]
#            offset = row_idx-target_index
#            row = current_matrix[target_index]
#            rolled_row = np.roll(row, offset)
#            reshaped_rolled_row = np.reshape(rolled_row, (-1, square_matrix_dim))
#            
#            # rebuild matrix based on order of ref matrix
#            rearranged_matrix = np.append(rearranged_matrix, reshaped_rolled_row, axis=0)
#        rearranged_matrix = np.reshape(rearranged_matrix, (-1, square_matrix_dim, square_matrix_dim))
#        rearranged_matrices = np.append(rearranged_matrices, rearranged_matrix, axis=0)
#        print('done rearranging and adding matrix ', mat_idx)
#    return rearranged_matrices

############################################################################################################
# MAKE NEW REORDERED MATRIX (SINGULAR) USING THE INFO LEARNED ABOVE

def reorder_matrix(h5, original_matrix, target_indices):
# make placeholder variable for new matrices and add the key matrix first
    square_matrix_dim = len(h5['dissimMatrix'][0])
    rearranged_matrix = np.empty((0,square_matrix_dim), float)    # for each matrix in matrix array
    # get corresponding idx list
    #go through each row and take the ccorresponding row from the key matrix
#    print(target_indices)
    for row_idx in range(square_matrix_dim):
        target_index = target_indices[row_idx]
        row = original_matrix[target_index]
#        offset = row_idx-target_index
#        rolled_row = np.roll(row, offset)
        rearranged_row = []
        for index in target_indices:
            rearranged_row.append(row[index])
#        print(row)
        rearranged_row_array = np.asarray(rearranged_row)
        reshaped_rearranged_row_array = np.reshape(rearranged_row_array, (-1, square_matrix_dim))
#        print(reshaped_rearranged_row_array)
        # rebuild matrix based on order of ref matrix
        rearranged_matrix = np.append(rearranged_matrix, reshaped_rearranged_row_array, axis=0)
    rearranged_matrix = rearranged_matrix
    return rearranged_matrix

##################################################################################################################################
#CONVERT 14DIM MATRIX TO 15 DIM BY REPEATING VALUES FOR THE NEGLECTED AUDIO
    
def generate_15dim_matrix(h5, idx, label_list):
    rearranged_dissim_matrix = h5['rearrangedDissimMatrix'][idx]
    dissim_matrix_mean = np.round(np.mean(rearranged_dissim_matrix), 2)
    for tech_idx, technique in enumerate(label_list):
        if technique in h5['participantInfo'][idx][28]:
            # calculate how many classes were before this. If class was 3, then there are 3 classes before it (0,1,2)
            # multiply number of previous classes (3) by number of examples per class (3)
            # as we want to be the NEXT number, but indices are always offset by -1 because of counting from 0
            # the NEXT and the -1 cancel each other out, therefore add or subract nothing from (numPrevClasses * examplesPerClass)
            neglected_audio_idx = (tech_idx)*3
            break
    new_15_matrix = np.zeros((15,15))
    # put values from old into new matrix, offsetting if after the critical index (neglected_audio_index)
    for row in range(15):
        if row < neglected_audio_idx:
            source_row = row
        elif row == neglected_audio_idx:
            continue
        else:
            source_row = row-1    
        for col in range(15):
            if col < neglected_audio_idx:
                source_col = col
            elif col == neglected_audio_idx:
                continue
            else:
                source_col = col-1
            new_15_matrix[row,col] = rearranged_dissim_matrix[source_row,source_col]
    # add in the averages
    for index in range(15):
        new_15_matrix[index,neglected_audio_idx] = dissim_matrix_mean
        new_15_matrix[neglected_audio_idx,index] = dissim_matrix_mean
    new_15_matrix[neglected_audio_idx,neglected_audio_idx] = 0
    return new_15_matrix

############################################################################################################
# REARRANGE ORDER OF MATRIX indices BY LABEL

def arrange_indices_by_labels(h5, experiment_idx, ref_audio_names):
    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    rearranged_ref_audio_names=[]
    target_indices=[]
    for label in label_list:
        for idx, entry in enumerate(ref_audio_names):
            if label in entry:
                rearranged_ref_audio_names.append(entry)
                target_indices.append(idx)
    return rearranged_ref_audio_names, target_indices

############################################################################################################
# print out visual display of heatmaps for rearranged dissim matrices


def display_dissim_heatmaps(dissim_matrix, ref_audio_names):
    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    shortened_ref_list = []
    for label_idx, label in enumerate(label_list):
        for ref_ind, referenceAudioName in enumerate(ref_audio_names):  
            if label in referenceAudioName:
                shortened_ref_list.append(label_list[label_idx][:3])
    fig, ax = plt.subplots(figsize=(20, 10))
    im = ax.imshow(dissim_matrix)
    ax.set_xticks(np.arange(len(ref_audio_names)))
    ax.set_yticks(np.arange(len(ref_audio_names)))
    ax.set_xticklabels(shortened_ref_list)
    ax.set_yticklabels(shortened_ref_list)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    # WRITE NUMBERS INTO CELLS
    for i in range(len(ref_audio_names)):
        for j in range(len(ref_audio_names)):
            text = ax.text(j, i, dissim_matrix[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Dissimilarity Matrix Display")
    fig.tight_layout()
    plt.show()
    return

############################################################################################################
# inspect contents of string list - if it matches with comparative list, assign a class number to it

def strings_to_classes(strings_list_to_convert, substring_list):
    class_list = []
    for string in strings_list_to_convert:
        for class_number, substring in enumerate(substring_list):
            if substring in string:
                class_list.append(class_number)
                break
    return class_list

############################################################################################################
# COMPARE ALL COLUMNS IN ARRAY AGAINST ONE ANOTHER AND RETURN ARRAY WITH CONTENT: COLUMN_A, COLUMN_B, R_VALUE, P_VALUE

def array_column_correlations(array, sig_thresh, test_name, show_plots=True):
    num_columns = array.shape[1]
    correlation_list = []
    for i in range(0, num_columns):
        for j in range(0, num_columns):
            if test_name=='pearsonr':
                r_val, p_val = pearsonr(array[:,i], array[:,j])
            elif test_name=='kendalltau':
                r_val, p_val = kendalltau(array[:,i], array[:,j])
            elif test_name=='spearmanr':
                r_val, p_val = spearmanr(array[:,i], array[:,j])
            correlation_list.append((i, j, r_val, p_val))
    correlation_array = np.asarray(correlation_list)
    print('Correlations: p<' +str(sig_thresh))
    correlation_sig_array = np.empty((0,4))
    for idx, i in enumerate(correlation_array):
        if i[2]<0.999:
            tester=3
            if abs(i[3])<=sig_thresh:
                if show_plots==True:
                    plt.scatter(array[:,int(i[0])], array[:,int(i[1])])
                    plt.show()
                    plt.close()
                    print('idx: ', idx, 'row: ', np.round(i, 2))
                reshaped_i = i.reshape(-1, len(i))
                correlation_sig_array = np.append(correlation_sig_array, reshaped_i, axis=0)
    return correlation_array, correlation_sig_array


############################################################################################################

def list_by_key_value(h5, key_ref, idx, value, key_use):
    # collect all sessions with target session name
    session_list=[]
    num_experiments = len(h5[key_ref])
    for experiment_idx in range(num_experiments):
        if type(value) == str:
            if value in h5[key_ref][experiment_idx][idx]:
                session_list.append(h5[key_use][experiment_idx])    
                print('Matching indices: ', experiment_idx)

        else:
            if h5['participantInfo'][experiment_idx][idx] == value:
                session_list.append(h5[key_use][experiment_idx])
                print('Matching indices: ', experiment_idx)
    return session_list

############################################################################################################

def indices_by_key_value(h5, key_ref, idx, value):
    # collect all sessions with target session name
    idx_list=[]
    num_experiments = len(h5[key_ref])
    for experiment_idx in range(num_experiments):
        if type(value) == str:
            if value in h5[key_ref][experiment_idx][idx]:
                idx_list.append(experiment_idx)    
#                print('Matching indices: ', experiment_idx)
        else:
            if h5['participantInfo'][experiment_idx][idx] == value:
                idx_list.append(experiment_idx)    
#                print('Matching indices: ', experiment_idx)
    return idx_list

############################################################################################################
# GET RATINGS FOR REPEATED PAGES AND GET THE RMSE FOR THE DIFFERENCE BETWEEN THEM


def evaluate_reliabiltiy(h5, experiment_idx):
    num_experiments = len(h5['participantInfo'])
    num_ratings = len(h5['dissimMatrix'][0])
    page_6_idx = page_14_idx = 100
    repeat_6_ratings = repeat_14_ratings = repeat_6_comp_audio = repeat_14_comp_audio = np.zeros(8)
    reliability_scores_list=[]
    represented_6_first=False

    # get the page index relevant to the repeated pages
    for idx in range(num_ratings):
        if h5['pageInfo'][experiment_idx][idx][2]=='6':
            page_6_idx = idx
        if h5['pageInfo'][experiment_idx][idx][2]=='14':
            page_14_idx = idx
    page_6_ref_name = h5['referenceAudioNames'][experiment_idx][page_6_idx]
    page_14_ref_name = h5['referenceAudioNames'][experiment_idx][page_14_idx]

    # find out which order repeat entries are saved as. Was page6 presented first?
    if h5['repeatReferenceAudioNames'][experiment_idx][0] == page_6_ref_name:
        # then assign variables for ratings and comparative audio
        repeat_6_ratings = h5['repeatParticipantRatings'][experiment_idx][0]
        repeat_14_ratings = h5['repeatParticipantRatings'][experiment_idx][1]
        repeat_6_comp_audio = h5['repeatComparativeAudioNames'][experiment_idx][0]
        repeat_14_comp_audio = h5['repeatComparativeAudioNames'][experiment_idx][1]
        represented_6_first = True
    else:
        repeat_6_ratings = h5['repeatParticipantRatings'][experiment_idx][1]
        repeat_14_ratings = h5['repeatParticipantRatings'][experiment_idx][0]   
        repeat_6_comp_audio = h5['repeatComparativeAudioNames'][experiment_idx][1]
        repeat_14_comp_audio = h5['repeatComparativeAudioNames'][experiment_idx][0]
    
    original_6_ratings = h5['participantRatings'][experiment_idx][page_6_idx]
    original_14_ratings = h5['participantRatings'][experiment_idx][page_14_idx]
    original_6_comp_audio = h5['comparativeAudioNames'][experiment_idx][page_6_idx]
    original_14_comp_audio = h5['comparativeAudioNames'][experiment_idx][page_14_idx]
    
    idx_for_ref_6_list = get_reordered_index_for_repeat(original_6_comp_audio, repeat_6_comp_audio)
    idx_for_ref_14_list = get_reordered_index_for_repeat(original_14_comp_audio, repeat_14_comp_audio)
    
    rmse_6 = get_rmse_between_repeats(original_6_ratings, repeat_6_ratings, idx_for_ref_6_list)
    rmse_14 = get_rmse_between_repeats(original_14_ratings, repeat_14_ratings, idx_for_ref_14_list)
        
    return rmse_6, rmse_14


############################################################################################################
    
def custom_kruskal(conditions_group):
    small_sample_sizes = False
    num_conditions=len(conditions_group)
    d_of_f = num_conditions - 1
    df_table_val = d_of_f - 1
    sample_sizes = []
    sample_means = []
    concatenated_samples = []
    
    for condition in conditions_group:
        sample_size = len(condition)
        sample_sizes.append(sample_size)
        sample_mean = sum(condition)/sample_size
        sample_means.append(sample_mean)
        concatenated_samples.extend(condition)
        
    ranked_samples = rankdata(concatenated_samples)
    sample_ranks = []
    total_participants = 0
    previous_end = 0
    for i in range(num_conditions):
        sample_rank = ranked_samples[previous_end:previous_end+sample_sizes[i]]
        sample_ranks.append(sample_rank)
        total_participants += sample_sizes[i]
        previous_end = previous_end+sample_sizes[i]
    sample_rank_totals = []
    sample_rank_totals_squared = []
    for sample_rank in sample_ranks:
        sample_rank_totals.append(sum(sample_rank))
        sample_rank_totals_squared.append(sum(sample_rank)*sum(sample_rank))
    
    sum_rank_total_squared = 0
    for i in range(len(sample_rank_totals_squared)):
        sum_rank_total_squared += sample_rank_totals_squared[i]/sample_sizes[i]
    
    for sample_size in sample_sizes:
        if sample_size < 5:
            small_sample_sizes = True
            print('small sample size found')
            
    if small_sample_sizes == False:             
        h_value = ( ( 12/(total_participants*(total_participants+1)) * sum_rank_total_squared ) - 3*(total_participants+1) )
        p_value = 1 - chi2.cdf(h_value, d_of_f)
    else:
        print('write some alternative code here for if your samples are too small!')
    return h_value, p_value

############################################################################################################

# function to calculate Cohen's d for independent samples
def cohend(d1, d2):
	# calculate the size of samples
	n1, n2 = len(d1), len(d2)
	# calculate the variance of the samples
	s1, s2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
	# calculate the pooled standard deviation
	s = math.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
	# calculate the means of the samples
	u1, u2 = np.mean(d1), np.mean(d2)
	# calculate the effect size
	return abs(u1 - u2) / s

############################################################################################################
# IF SAMPLES ARE LESS THAN 20, MANUALLY CHECK U VALUE AGAINST THE APPROPRIATE TABLE
def custom_mann_whit(conditions_group): 
    condition1 = np.asarray(conditions_group[0])
    condition2 = np.asarray(conditions_group[1])
    # depending on removed indices, <=30 sample sizes per condition
    sample_a_size = len(condition1)
    sample_b_size = len(condition2)
    sample_a_mean = np.mean(condition1)
    sample_b_mean = np.mean(condition2)


    if type(conditions_group[0])==np.ndarray:
        concatenated_samples = conditions_group[0].tolist()
    else:       
        concatenated_samples = conditions_group[0].copy()
    for value in conditions_group[1]:
        concatenated_samples.append(value)
    ranked_samples = rankdata(concatenated_samples)
    sample_a_ranked = ranked_samples[:sample_a_size]
    sample_b_ranked = ranked_samples[sample_a_size:]
    sample_a_ranked_mean = np.round(np.mean(sample_a_ranked), 2)
    sample_b_ranked_mean = np.round(np.mean(sample_b_ranked), 2)
    
    sample_a_summed_ranks=0
    for i in sample_a_ranked:
        sample_a_summed_ranks += i
    
    u_value = sample_a_size*sample_b_size+(sample_a_size*(sample_a_size + 1)/2) - sample_a_summed_ranks
    u_value_prime = sample_a_size*sample_b_size - u_value
    
    if u_value_prime < u_value:
        u_value = u_value_prime
        
    normal_mean = (sample_a_size*sample_b_size)/2
    normal_std = math.sqrt((sample_a_size*sample_b_size*(sample_a_size + sample_b_size + 1))/12)
    z_value = (u_value - normal_mean)/normal_std
    
    # effect size for interval data?
    
    classic_cohen_d = round(cohend(condition1,condition2), 3) # rounded to 3 decimal points
    
    ordinal_cohen_d = z_value/math.sqrt(sample_a_size*sample_b_size) # uncertain if this is the right way to do it - double-check

#    print(sample_a_mean, sample_b_mean)
    return sample_a_mean, sample_b_mean, abs(z_value), ordinal_cohen_d, u_value, sample_a_size, sample_b_size


############################################################################################################

# self describing function name!

def para_multiple_variables_ttest(main_group, sig_thresh):
    t_test_group = []
    annotated_t_test_group = np.empty((0,16))
    z_value_threshold = stats.norm.ppf(1 - sig_thresh)
    for variables_group_idx, variables_group in enumerate(main_group):
        if variables_group_idx==0:
            continue
        for conditions_group_idx, conditions_group in enumerate(variables_group):
            num_conditions = len(conditions_group)
            d_of_f = num_conditions-1

            condition_pairs = list(itertools.combinations(range(num_conditions), 2))
            # adjust sig levels
            bonferri_corrected_sig_thresh = sig_thresh/d_of_f
            bonferri_corrected_z_threshold = stats.norm.ppf(1 - bonferri_corrected_sig_thresh)
            # go through every combination of conditions
            for con_pair_idx, (condition_a, condition_b) in enumerate(condition_pairs):
                
                condition_array_a = np.asarray(conditions_group[condition_a])
                condition_array_b = np.asarray(conditions_group[condition_b])
                sample_a_mean = np.mean(condition_array_a)
                sample_b_mean = np.mean(condition_array_b)
                sample_a_size = len(condition_array_a)
                sample_b_size = len(condition_array_b)
                
                classic_cohen_d = round(cohend(condition_array_a, condition_array_b), 3)
                t_stat, p_value = ttest_ind(condition_array_a, condition_array_b)   
                
#                    print('z_value for first condition pair {0} is {1}'.format(condition_pairs[con_pair_idx], z_value))
                if p_value < (bonferri_corrected_sig_thresh):
#                    actual_p_val = 1 - stats.norm.cdf(z_value)
#                    print('sample_a_size, sample_b_size', sample_a_size, sample_b_size)  
                    
                    mean_difference = np.round(abs(sample_a_mean-sample_b_mean), 2)
                    if sample_a_mean > sample_b_mean:
                        higher_con_tuple = (condition_a, sample_a_mean)
                        lower_con_tuple = (condition_b, sample_b_mean)
                    else:
                        higher_con_tuple = (condition_b, sample_b_mean)
                        lower_con_tuple = (condition_a, sample_a_mean)

                    print('variables_group_idx: ', variables_group_idx,
                                 'conditions_group_idx: ', conditions_group_idx,
                                 'cond_causing_increase, sample_mean: ', higher_con_tuple,
                                 'cond_causing_decrease, sample_mean: ',lower_con_tuple,
                                 'Mean increase: ', mean_difference,
                                 'actual_p_val: ', round(p_value, 4),
                                 'p value thresh: ', round(bonferri_corrected_sig_thresh, 4),
                                 'classic_cohen_d:15 ', classic_cohen_d
                                 )
                    readings = [variables_group_idx,
                                 conditions_group_idx,
                                 higher_con_tuple,
                                 lower_con_tuple,
                                 round(mean_difference, 4),
                                 round(p_value, 4),
                                 round(bonferri_corrected_sig_thresh, 4),
                                 classic_cohen_d]
                    annotated_readings = np.array(['variables_group_idx:1 ', variables_group_idx,
                                                   'conditions_group_idx:3 ', conditions_group_idx,
                                                   'cond_causing_increase, sample_mean:5 ', higher_con_tuple,
                                                   'cond_causing_decrease, sample_mean:7 ',lower_con_tuple,
                                                   'Mean increase:9 ', mean_difference,
                                                   'actual_p_val:11 ', round(p_value, 4),
                                                   'p value thresh:13 ', round(bonferri_corrected_sig_thresh, 4),
                                                   'classic_cohen_d:15 ', classic_cohen_d])
                    t_test_group.append(readings)
                    annotated_readings = annotated_readings.reshape(-1, len(annotated_readings))
                    annotated_t_test_group = np.append(annotated_t_test_group, annotated_readings, axis=0)
    
    return annotated_t_test_group, t_test_group

############################################################################################################

# self describing function name!

def nonpara_multiple_variables_test(main_group, sig_thresh):
    custom_mann_whit_group = []
    annotated_custom_mann_whit_group = np.empty((0,20))
    z_value_threshold = stats.norm.ppf(1 - sig_thresh)
    for variables_group_idx, variables_group in enumerate(main_group):
        for conditions_group_idx, conditions_group in enumerate(variables_group):
            num_conditions = len(conditions_group)
            d_of_f = num_conditions-1
            # check which group to know how many conditions to provide to kruskal object
    #        print(variables_group_idx, conditions_group_idx)
    #        if num_conditions==12:
    #            # depending on removed indices, <=5 sample sizes per condition
    #            print(variables_group_idx, class_pairs[conditions_group_idx], '12 CONDITIONS WITH ONLY 5 PARTICIPANTS - GREENE PROVIDES NO METHOF FOR THIS')
            
            if num_conditions>2 and num_conditions<10:
                # GET H AND P VALUES
    #            take the stats number and look it up on the chi-square table (Tabe E in the book)
    #            scipy_kruskal_results = stats.kruskal(conditions_group[0], conditions_group[1], conditions_group[2], conditions_group[3])
    #            if kruskal_results[0]>critical_vals_for_05[df_table_val]:
    #                if kruskal_results[1]<one_tailed_sig_thresh:             
                h_value, p_value = custom_kruskal(conditions_group)
                # chisquare function takes h_value and degrees of freedom returns a p value
                if p_value < sig_thresh:
    #                reading = ('Rating Group Idx: ', variables_group_idx, 'class_pairs: ', class_pairs[conditions_group_idx], 'kruskal_results: ', h_value, p_value)
    ##                print(reading)
    #                custom_kruskal_group.append(reading)
                    # if there is a significant difference, inspect this firuther with mann whitney and adjusted signficance to balance likelihood against multiple comparisons
                    condition_pairs = list(itertools.combinations(range(num_conditions), 2))
                    # adjust sig levels
                    bonferri_corrected_sig_thresh = sig_thresh/d_of_f
                    bonferri_corrected_z_threshold = stats.norm.ppf(1 - bonferri_corrected_sig_thresh)
                    # go through every combination of conditions
                    for con_pair_idx, (condition_a, condition_b) in enumerate(condition_pairs):
                        contrast_conditions = (conditions_group[condition_a], conditions_group[condition_b])
                        sample_a_mean, sample_b_mean, z_value, ordinal_cohen_d, u_value, sample_a_size, sample_b_size = custom_mann_whit(contrast_conditions)
                        
                        
    #                    print('z_value for first condition pair {0} is {1}'.format(condition_pairs[con_pair_idx], z_value))
                        if z_value < (-bonferri_corrected_z_threshold) or z_value > (bonferri_corrected_z_threshold):
                            actual_p_val = 1 - stats.norm.cdf(z_value)
                            print('sample_a_size, sample_b_size', sample_a_size, sample_b_size)  
                            
                            mean_difference = np.round(abs(sample_a_mean-sample_b_mean), 2)
                            if sample_a_mean > sample_b_mean:
                                higher_con_tuple = (condition_a, sample_a_mean)
                                lower_con_tuple = (condition_b, sample_b_mean)
                            else:
                                higher_con_tuple = (condition_b, sample_b_mean)
                                lower_con_tuple = (condition_a, sample_a_mean)

                            print('variables_group_idx: ', variables_group_idx,
                                         'conditions_group_idx: ', conditions_group_idx,
                                         'condition_causing_increase: ', higher_con_tuple,
                                         'condition_causing_decrease: ',lower_con_tuple,
                                         'Mean increase: ', mean_difference,
                                         'Effect size: ', ordinal_cohen_d,
                                         'u-value: ', u_value,
                                         'z value: ', round(z_value, 4),
                                         'actual_p_val: ', round(actual_p_val, 4),
                                         'p value thresh: ', round(bonferri_corrected_sig_thresh, 4)
                                         )
                            readings = [variables_group_idx,
                                         conditions_group_idx,
                                         higher_con_tuple,
                                         lower_con_tuple,
                                         round(mean_difference, 4),
                                         ordinal_cohen_d,
                                         u_value,
                                         round(z_value, 4),
                                         round(actual_p_val, 4),
                                         round(bonferri_corrected_sig_thresh, 4)]
                            annotated_readings = np.array(['variables_group_idx:1 ', variables_group_idx,
                                                           'conditions_group_idx:3 ', conditions_group_idx,
                                                           'condition_causing_increase:5 ', higher_con_tuple,
                                                           'condition_causing_decrease:7 ',lower_con_tuple,
                                                           'Mean increase:9 ', mean_difference,
                                                           'Effect size:11 ', ordinal_cohen_d,
                                                           'u-value: ', u_value,
                                                           'z value:13 ', round(z_value, 4),
                                                           'actual_p_val:15 ', round(actual_p_val, 4),
                                                           'p value thresh:17 ', round(bonferri_corrected_sig_thresh, 4)])
                            custom_mann_whit_group.append(readings)
                            annotated_readings = annotated_readings.reshape(-1, len(annotated_readings))
                            annotated_custom_mann_whit_group = np.append(annotated_custom_mann_whit_group, annotated_readings, axis=0)
            
            elif num_conditions==2:
#                print(contrast_conditions)
                # same contrast condition info comes in THE FIRST TIME, but is calculated differently in cmw2 than in cmw2
                sample_a_mean, sample_b_mean, z_value, ordinal_cohen_d, u_value, sample_a_size, sample_b_size = custom_mann_whit(conditions_group)
                if z_value < (-z_value_threshold) or z_value > (z_value_threshold):
                    actual_p_val = 1 - stats.norm.cdf(z_value)
                    print('sample_a_size, sample_b_size', sample_a_size, sample_b_size)  
                    
                    mean_difference = abs(sample_a_mean-sample_b_mean)
                    if sample_a_mean > sample_b_mean:
                        higher_con_tuple = (0, sample_a_mean)
                        lower_con_tuple = (1, sample_b_mean)
                    else:
                        higher_con_tuple = (1, sample_b_mean)
                        lower_con_tuple = (0, sample_a_mean)
                    
                    print('variables_group_idx: ', variables_group_idx,
                                 'conditions_group_idx: ', conditions_group_idx,
                                 'condition_causing_increase: ', higher_con_tuple,
                                 'condition_causing_decrease: ',lower_con_tuple,
                                 'Mean increase: ', mean_difference,
                                 'Effect size: ', ordinal_cohen_d,
                                 'u-value: ', u_value,
                                 'z value: ', round(z_value, 4),
                                 'actual_p_val: ', round(actual_p_val, 4),
                                 'p value thresh: ', round(sig_thresh, 4),
                                 )   
                    readings = [variables_group_idx,
                                 conditions_group_idx,
                                 higher_con_tuple,
                                 lower_con_tuple,
                                 round(mean_difference, 4),
                                 ordinal_cohen_d,
                                 u_value,
                                 round(z_value, 4),
                                 round(actual_p_val, 4),
                                 round(sig_thresh, 4)]
                    annotated_readings = np.array(['variables_group_idx:1 ', variables_group_idx,
                                                   'conditions_group_idx:3 ', conditions_group_idx,
                                                   'condition_causing_increase:5 ', higher_con_tuple,
                                                   'condition_causing_decrease:7 ',lower_con_tuple,
                                                   'Mean increase:9 ', mean_difference,
                                                   'Effect size:11 ', ordinal_cohen_d,
                                                   'u-value: ', u_value,
                                                   'z value:13 ', round(z_value, 4),
                                                   'actual_p_val:15 ', round(actual_p_val, 4),
                                                   'p value thresh:17 ', round(sig_thresh, 4)])
                    custom_mann_whit_group.append(readings)
                    annotated_readings = annotated_readings.reshape(-1, len(annotated_readings))
                    annotated_custom_mann_whit_group = np.append(annotated_custom_mann_whit_group, annotated_readings, axis=0)
    
    return annotated_custom_mann_whit_group, custom_mann_whit_group

############################################################################################################

# self describing function name!

def nonpara_single_variable_test(main_group, sig_thresh):
    custom_mann_whit_group = []
    annotated_custom_mann_whit_group = np.empty((0,18))
    z_value_threshold = stats.norm.ppf(1 - sig_thresh)
    for conditions_group_idx, conditions_group in enumerate(main_group):
        num_conditions = len(conditions_group)
        d_of_f = num_conditions-1
        # check which group to know how many conditions to provide to kruskal object
#        print(variables_group_idx, conditions_group_idx)
#        if num_conditions==12:
#            # depending on removed indices, <=5 sample sizes per condition
#            print(variables_group_idx, class_pairs[conditions_group_idx], '12 CONDITIONS WITH ONLY 5 PARTICIPANTS - GREENE PROVIDES NO METHOF FOR THIS')
        
        if num_conditions>2 and num_conditions<10:
            # GET H AND P VALUES
#            take the stats number and look it up on the chi-square table (Tabe E in the book)
#            scipy_kruskal_results = stats.kruskal(conditions_group[0], conditions_group[1], conditions_group[2], conditions_group[3])
#            if kruskal_results[0]>critical_vals_for_05[df_table_val]:
#                if kruskal_results[1]<one_tailed_sig_thresh:             
            h_value, p_value = custom_kruskal(conditions_group)
            # chisquare function takes h_value and degrees of freedom returns a p value
            if p_value < sig_thresh:
#                reading = ('Rating Group Idx: ', variables_group_idx, 'class_pairs: ', class_pairs[conditions_group_idx], 'kruskal_results: ', h_value, p_value)
##                print(reading)
#                custom_kruskal_group.append(reading)
                # if there is a significant difference, inspect this firuther with mann whitney and adjusted signficance to balance likelihood against multiple comparisons
                condition_pairs = list(itertools.combinations(range(num_conditions), 2))
                # adjust sig levels
                bonferri_corrected_sig_thresh = sig_thresh/d_of_f
                bonferri_corrected_z_threshold = stats.norm.ppf(1 - bonferri_corrected_sig_thresh)
                # go through every combination of conditions
                for con_pair_idx, (condition_a, condition_b) in enumerate(condition_pairs):
                    contrast_conditions = (conditions_group[condition_a], conditions_group[condition_b])
                    sample_a_mean, sample_b_mean, z_value, ordinal_cohen_d, u_value, sample_a_size, sample_b_size = custom_mann_whit(contrast_conditions)
                    
#                    print('z_value for first condition pair {0} is {1}'.format(condition_pairs[con_pair_idx], z_value))
                    if z_value < (-bonferri_corrected_z_threshold) or z_value > (bonferri_corrected_z_threshold):
                        actual_p_val = 1 - stats.norm.cdf(z_value)
                        print('sample_a_size, sample_b_size', sample_a_size, sample_b_size)  
                        
                        mean_difference = abs(sample_a_mean-sample_b_mean)
                        if sample_a_mean > sample_b_mean:
                            higher_con_tuple = (condition_a, sample_a_mean)
                            lower_con_tuple = (condition_b, sample_b_mean)
                        else:
                            higher_con_tuple = (condition_b, sample_b_mean)
                            lower_con_tuple = (condition_a, sample_a_mean)

                        print(
                                     'conditions_group_idx: ', conditions_group_idx,
                                     'condition_causing_increase: ', higher_con_tuple,
                                     'condition_causing_decrease: ',lower_con_tuple,
                                     'Mean increase: ', mean_difference,
                                     'Effect size: ', ordinal_cohen_d,
                                     'u-value: ', u_value,
                                     'z value: ', round(z_value, 4),
                                     'actual_p_val: ', round(actual_p_val, 4),
                                     'p value thresh: ', round(bonferri_corrected_sig_thresh, 4)
                                     )
                        readings = [
                                     conditions_group_idx,
                                     higher_con_tuple,
                                     lower_con_tuple,
                                     round(mean_difference, 4),
                                     ordinal_cohen_d,
                                     u_value,
                                     round(z_value, 4),
                                     round(actual_p_val, 4),
                                     round(bonferri_corrected_sig_thresh, 4)]
                        annotated_readings = np.array([
                                                       'conditions_group_idx:1 ', conditions_group_idx,
                                                       'condition_causing_increase:3 ', higher_con_tuple,
                                                       'condition_causing_decrease:5 ',lower_con_tuple,
                                                       'Mean increase:7 ', mean_difference,
                                                       'Effect size:9 ', ordinal_cohen_d,
                                                       'u-value: ', u_value,
                                                       'z value:11 ', round(z_value, 4),
                                                       'actual_p_val:13 ', round(actual_p_val, 4),
                                                       'p value thresh:15 ', round(bonferri_corrected_sig_thresh, 4)])
                        custom_mann_whit_group.append(readings)
                        annotated_readings = annotated_readings.reshape(-1, len(annotated_readings))
                        annotated_custom_mann_whit_group = np.append(annotated_custom_mann_whit_group, annotated_readings, axis=0)
        
        elif num_conditions==2:
#                print(contrast_conditions)
            # same contrast condition info comes in THE FIRST TIME, but is calculated differently in cmw2 than in cmw2
            sample_a_mean, sample_b_mean, z_value, ordinal_cohen_d, u_value, sample_a_size, sample_b_size = custom_mann_whit(conditions_group)
            if z_value < (-z_value_threshold) or z_value > (z_value_threshold):
                actual_p_val = 1 - stats.norm.cdf(z_value)
                print('sample_a_size, sample_b_size', sample_a_size, sample_b_size)  
                
                mean_difference = abs(sample_a_mean-sample_b_mean)
                if sample_a_mean > sample_b_mean:
                    higher_con_tuple = (0, sample_a_mean)
                    lower_con_tuple = (1, sample_b_mean)
                else:
                    higher_con_tuple = (1, sample_b_mean)
                    lower_con_tuple = (0, sample_a_mean)
                
                print(
                             'conditions_group_idx: ', conditions_group_idx,
                             'condition_causing_increase: ', higher_con_tuple,
                             'condition_causing_decrease: ',lower_con_tuple,
                             'Mean increase: ', mean_difference,
                             'Effect size: ', ordinal_cohen_d,
                             'u-value: ', u_value,
                             'z value: ', round(z_value, 4),
                             'actual_p_val: ', round(actual_p_val, 4),
                             'p value thresh: ', round(sig_thresh, 4)
                             )   
                readings = [
                             conditions_group_idx,
                             higher_con_tuple,
                             lower_con_tuple,
                             round(mean_difference, 4),
                             ordinal_cohen_d,
                             u_value,
                             round(z_value, 4),
                             round(actual_p_val, 4),
                             round(sig_thresh, 4)]
                annotated_readings = np.array([
                                                   'conditions_group_idx:1 ', conditions_group_idx,
                                                   'condition_causing_increase:3 ', higher_con_tuple,
                                                   'condition_causing_decrease:5 ',lower_con_tuple,
                                                   'Mean increase:7 ', mean_difference,
                                                   'Effect size:9 ', ordinal_cohen_d,
                                                   'u-value: ', u_value,
                                                   'z value:11 ', round(z_value, 4),
                                                   'actual_p_val:13 ', round(actual_p_val, 4),
                                                   'p value thresh:15 ', round(bonferri_corrected_sig_thresh, 4)])
                custom_mann_whit_group.append(readings)
                annotated_readings = annotated_readings.reshape(-1, len(annotated_readings))
                annotated_custom_mann_whit_group = np.append(annotated_custom_mann_whit_group, annotated_readings, axis=0)
    
    return annotated_custom_mann_whit_group, custom_mann_whit_group