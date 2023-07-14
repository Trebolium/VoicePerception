import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.manifold import MDS

from utils import bytes_to_str  

def display_dissim_heatmaps(dissim_matrix, ref_audio_names, idx, save_plot=False):
    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    shortened_ref_list = []
    for label_idx, label in enumerate(label_list):
        for ref_ind, referenceAudioName in enumerate(ref_audio_names):  
            referenceAudioName = bytes_to_str(referenceAudioName)
            if label in referenceAudioName:
                shortened_ref_list.append(label_list[label_idx][:3])
    fig, ax = plt.subplots(figsize=(10, 8))
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
    title=f"Dissimilarity Matrix of Participant {idx}"
    ax.set_title(title)
    fig.tight_layout()
    if save_plot:
        dir_name = 'dissim_matrices'
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        plt.savefig(os.path.join(dir_name, title))
    plt.show()
    plt.close()


def summarize_cond_grouped_ratings_mds(
    cond_grouped_ratings,
    statistical_summary,
    condition_labels,
    class_pairs,
    label_list,
    mds_dim=2,
    show_plot=True):
    
    classpair_groups = []
    mds_groups = []

    for class_pair_key in cond_grouped_ratings.keys():
    #     print('class_pair_key', class_pair_key)
        cond_group = []

        # key_2s relate to the class-pair variable names ((0,1), (0,2) etc.)
        for cond_specific_key in cond_grouped_ratings[class_pair_key].keys():
#             print('cond_specific_key', cond_specific_key)

            # key_3 relates to which condition specifically can be accessed with it
            avg_vals_list = cond_grouped_ratings[class_pair_key][cond_specific_key]
            if statistical_summary == 'mean':
                value_summary = np.mean(avg_vals_list)
            elif statistical_summary == 'median':
                value_summary = np.median(avg_vals_list)
            else:
                NotImplementedError
            cond_group.append(value_summary)
        classpair_groups.append(np.asarray(cond_group))

    classpair_specific_group_array = np.asarray(classpair_groups)
    # print('classpair_specific_group_array,shape', classpair_specific_group_array.shape)

    for condition_int in range(classpair_specific_group_array.shape[1]):
        class_dissim_matrix = np.empty((5,5))
        conditioned_variables_array = classpair_specific_group_array[:,condition_int]
        for classpair_idx in range(len(class_pairs)):
            coord_1, coord_2 = class_pairs[classpair_idx]
            class_dissim_matrix[coord_1, coord_2] = conditioned_variables_array[classpair_idx]

        for i in range(5):
            for j in range(5):
                class_dissim_matrix[j,i] = class_dissim_matrix[i,j]

        title = condition_labels[condition_int]
        mds_coords = mds_from_dissim_matrix(class_dissim_matrix, label_list, mds_dim, show_plot=show_plot, title=title)
        mds_groups.append(mds_coords)
        
    return np.asarray(classpair_groups), np.asarray(mds_groups)


def mds_from_dissim_matrix(dissim_matrix, ref_list, dimensions, show_plot=True, title='mds_plot',save_plot=True):
    np.random.seed(19680801)
#    label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    label_dict = {'straight':0, 'belt':1, 'breathy':2, 'fry':3, 'vibrato':4}
    color_list = ['r', 'g', 'b', 'c', 'm']

#    print(dissim_matrix.shape)
    embedding = MDS(n_components=dimensions, metric=False, dissimilarity='precomputed')
    exp_transfomed = embedding.fit_transform(dissim_matrix)
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
        # plt.title(title)
        plt.legend(label_dict)
        if save_plot:
            dir_name = 'mds_plots'
            if not os.path.exists(dir_name):
                os.mkdir(dir_name)
            plt.savefig(os.path.join(dir_name, title))
        plt.show()
        plt.close()
    return exp_transfomed