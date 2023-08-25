from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score
from kneed import KneeLocator

from utils import bytes_to_str, strings_to_classes


def get_elbow_array(h5, indices, k_range, file_name, show_plot, s_param, save_plot=False):

    num_experiments = len(indices)
    elbows=np.empty(num_experiments)
    for idx_idx, idx in enumerate(indices):
        rearranged_dissim_matrix = h5['dissim_matrix'][idx]
        dissim_matrix = np.around(rearranged_dissim_matrix, 2)    
        cluster_scores = cluster_scoring(idx, dissim_matrix, k_range, show_plot=show_plot)
        kneedle = KneeLocator(k_range, cluster_scores, S=s_param, curve='convex', direction='decreasing')
    #    print('idx: ', idx,'elbow_dim: ', kneedle.elbow)
        elbows[idx_idx] = kneedle.elbow
    if show_plot:
        elbow_bar_solutions = []
        # get range between (inclusive means adding +1) min and max
        cluster_range=range(int(np.min(elbows)),int(np.max(elbows)+1))
        for entry in cluster_range:
            cluster_num = entry
            total_sum_per_cluster_num = np.count_nonzero(elbows==cluster_num)
            elbow_bar_solutions.append(total_sum_per_cluster_num)
            
        plt.bar(cluster_range, elbow_bar_solutions)
        # plt.title(file_name)
        plt.xticks(k_range)
#        plt.yticks(range(0,40,2))
        plt.tight_layout()
        if save_plot:
            subdir = 'kMeans elbow bar graph'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            plt.savefig(os.path.join(subdir, file_name))
        plt.show()
        plt.close()
#        print('With S= ' +str(s_level))
    return elbows


def cluster_scoring(idx, dissim_matrix, k_range, show_plot=True, save_plot=False):

    sse_list=[]
    sse = kmeans_sse_from_arrays(k_range, dissim_matrix)
    if show_plot==True:
        fig_1 = plt.figure(1, figsize=(6.4, 4.8))
        plt.plot(k_range, sse)
#        # plt.title('SSE per k number for Exp ' +str(idx))
        plt.xlabel('Dimensions for k')
        plt.ylabel('SSE loss')
        if save_plot:
            subdir = 'kMeans_SSEs_default_setting'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            plt.savefig(os.path.join(subdir, f'Participant {str(idx)}'))
        plt.show()
        plt.close()
    return sse


def kmeans_sse_from_arrays(k_range, arrays):
    sum_squared_error = []
    for k_dim in k_range:
        km = KMeans(n_clusters=k_dim)
        km.fit_predict(arrays)
        sum_squared_error.append(km.inertia_)
    return sum_squared_error


def strings_to_classes(strings_list_to_convert, substring_list):
    class_list = []
    for string in strings_list_to_convert:
        string = bytes_to_str(string)
        for class_number, substring in enumerate(substring_list):
            substring = bytes_to_str(substring) 
            if substring in string:
                class_list.append(class_number)
                break
    return class_list


def get_silhouette_accuracy(dataset, cluster_algorithm, num_clusters, class_list, score_type):
    """
        Gets the sillhouette and accuracy score for a clustering label predictions on a given dataset
    """
    if cluster_algorithm == 'kmeans':
        cluster_model = KMeans(n_clusters=num_clusters)
    elif cluster_algorithm == 'agglomerative':
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters, affinity = 'euclidean', linkage='ward')
    
    cluster_group_predictions = cluster_model.fit_predict(dataset) 
    if score_type == 'silhouette':
        score = silhouette_score(dataset, cluster_group_predictions, metric='euclidean')
    elif score_type == 'accuracy':
        score = adjusted_rand_score(cluster_group_predictions, class_list)
    else:
        raise NotImplementedError
    
    return round(score, 2)


def find_best_k_values(h5, indices, condition_name, k, cluster_algorithm, score_type, show_plot=True, save_plot=False):

    if type(k) == range:
        
        if cluster_algorithm == 'kmeans': k_offset = 2   
        elif cluster_algorithm == 'agglomerative': k_offset = 2
        k_range = k
    elif type(k) == int:
        k_range = range(k, k+1)
        k_offset = k
    else: raise NotImplementedError

    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    best_k_from_scores = []
    scores_for_all_ks = np.zeros( (len(indices), len(k_range)) )

    # go through each participant dataset
    for idx_idx, idx in enumerate(indices):

        scores = []
        rearranged_dissim_matrix = h5['dissim_matrix'][idx]
        rearranged_ref_audio = h5['rearrangedReferenceAudioNames'][idx]
        rearranged_class_list = strings_to_classes(rearranged_ref_audio, label_list)


#         pdb.set_trace()
        for num_clusters in k_range:

            score = get_silhouette_accuracy(rearranged_dissim_matrix,
                                            cluster_algorithm,
                                            num_clusters,
                                            rearranged_class_list,
                                            score_type)
            scores.append(score)

        scores = np.asarray(scores)
        scores_for_all_ks[idx_idx, ...] = scores
        best_k_from_scores.append(np.argmax(scores)+k_offset)

    # Use Counter to count the occurrences of each element
    counter = Counter(best_k_from_scores)

    # Find the most common element and its count
    most_common_element = counter.most_common(1)[0][0]
    scores_using_best_k = scores_for_all_ks[:, most_common_element-k_offset]
    
    # print bar graphs of best k scores
    if show_plot:
        # turn best_k_from_silhouettes into bar graph suitable data
        # os.makedirs(f'{cluster_algorithm}_{score_type}_distributions/', exist_ok=True)
        title = f'{condition_name}_{cluster_algorithm}_{score_type}_distributions'
        best_k_hist = []
        cluster_solution_range = range(int(np.min(best_k_from_scores)),int(np.max(best_k_from_scores)+1))
        for cluster_num in cluster_solution_range:
            total_sum_per_cluster_num = np.count_nonzero(np.asarray(best_k_from_scores)==cluster_num)
            best_k_hist.append(total_sum_per_cluster_num)
            
        plt.bar(cluster_solution_range, best_k_hist)
        # plt.title(title)
        plt.xticks(k_range)
        plt.xlabel('Number of clusters (k)')
        plt.tight_layout()
        if save_plot:
            subdir = 'cluster_score_distributions'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            plt.savefig(os.path.join(subdir, title))
        plt.show()
        plt.close()

    return scores_using_best_k, best_k_from_scores


def get_silhouette_accuracy(dataset, cluster_algorithm, num_clusters, class_list, score_type):
    """
        Gets the sillhouette and accuracy score for a clustering label predictions on a given dataset
    """
    if cluster_algorithm == 'kmeans':
        cluster_model = KMeans(n_clusters=num_clusters)
    elif cluster_algorithm == 'agglomerative':
        cluster_model = AgglomerativeClustering(n_clusters=num_clusters, affinity = 'euclidean', linkage='ward')
    
    cluster_group_predictions = cluster_model.fit_predict(dataset) 
    if score_type == 'silhouette':
        score = silhouette_score(dataset, cluster_group_predictions, metric='euclidean')
    elif score_type == 'accuracy':
        score = adjusted_rand_score(cluster_group_predictions, class_list)
    else:
        raise NotImplementedError
    
    return round(score, 2)  


def find_best_k_values(h5, indices, condition_name, k, cluster_algorithm, score_type, show_plot=True, save_plot=False):

    if type(k) == range:
        
        if cluster_algorithm == 'kmeans': k_offset = 2   
        elif cluster_algorithm == 'agglomerative': k_offset = 2
        k_range = k
    elif type(k) == int:
        k_range = range(k, k+1)
        k_offset = k
    else: raise NotImplementedError

    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    best_k_from_scores = []
    scores_for_all_ks = np.zeros( (len(indices), len(k_range)) )

    # go through each participant dataset
    for idx_idx, idx in enumerate(indices):

        scores = []
        rearranged_dissim_matrix = h5['dissim_matrix'][idx]
        rearranged_ref_audio = h5['rearrangedReferenceAudioNames'][idx]
        rearranged_class_list = strings_to_classes(rearranged_ref_audio, label_list)


#         pdb.set_trace()
        for num_clusters in k_range:

            score = get_silhouette_accuracy(rearranged_dissim_matrix,
                                            cluster_algorithm,
                                            num_clusters,
                                            rearranged_class_list,
                                            score_type)
            scores.append(score)

        scores = np.asarray(scores)
        scores_for_all_ks[idx_idx, ...] = scores
        best_k_from_scores.append(np.argmax(scores)+k_offset)

    # Use Counter to count the occurrences of each element
    counter = Counter(best_k_from_scores)

    # Find the most common element and its count
    most_common_element = counter.most_common(1)[0][0]
    scores_using_best_k = scores_for_all_ks[:, most_common_element-k_offset]
    
    # print bar graphs of best k scores
    if show_plot:
        # turn best_k_from_silhouettes into bar graph suitable data
        # os.makedirs(f'{cluster_algorithm}_{score_type}_distributions/', exist_ok=True)
        title = f'{condition_name}_{cluster_algorithm}_{score_type}_distributions'
        plot_title = f'Best {score_type} for {cluster_algorithm} solutions'
        best_k_hist = []
        cluster_solution_range = range(int(np.min(best_k_from_scores)),int(np.max(best_k_from_scores)+1))
        for cluster_num in cluster_solution_range:
            total_sum_per_cluster_num = np.count_nonzero(np.asarray(best_k_from_scores)==cluster_num)
            best_k_hist.append(total_sum_per_cluster_num)
            
        plt.bar(cluster_solution_range, best_k_hist)
        plt.title(plot_title)
        plt.xticks(k_range)
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Number of best scores')
        plt.tight_layout()
        if save_plot:
            subdir = 'cluster_score_distributions'
            if not os.path.exists(subdir):
                os.mkdir(subdir)
            plt.savefig(os.path.join(subdir, title))
        plt.show()
        plt.close()

    return scores_using_best_k, best_k_from_scores


def get_scores_for_cond_indices(h5_ram, indices_list, condition_name, k, show_plots=True):
    print(f'{condition_name.upper()} \n')
    agg_sil_score, best_agg_sil = find_best_k_values(h5=h5_ram,
                                 indices=indices_list,
                                 condition_name=condition_name,
                                 k=k,
                                 cluster_algorithm='agglomerative',
                                 score_type='silhouette',
                                 show_plot=show_plots,
                                 save_plot=True)
    agg_acc_score, best_agg_acc = find_best_k_values(h5=h5_ram,
                                 indices=indices_list,
                                 condition_name=condition_name,
                                 k=k,
                                 cluster_algorithm='agglomerative',
                                 score_type='accuracy',
                                 show_plot=show_plots,
                                 save_plot=True)

    return agg_sil_score, best_agg_sil, agg_acc_score, best_agg_acc