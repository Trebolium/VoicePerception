import ast
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb


def indices_by_key_value(h5, key_ref, idx, value):
    # collect all sessions with target session name
    idx_list=[]
    num_experiments = len(h5[key_ref])
    for experiment_idx in range(num_experiments):
        session_name = h5[key_ref][experiment_idx][idx]
        session_name = bytes_to_str(session_name)
        value = bytes_to_str(value)
        if value in session_name:
            idx_list.append(experiment_idx)    

    return idx_list


def list_to_csvfile(list_of_rows, fp):
    with open(fp, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(list_of_rows)


def bytes_to_str(v):
    if type(v) == bytes or type(v) == np.bytes_:
        v = v.decode('utf-8')
    return v


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


def diagonal_array_handling(diagonal_arrays):
    """Takes list of 'list of strings', converts strings to floats, sums floats, return list of summed floats"""
    if type(diagonal_arrays[0]) == bytes or type(diagonal_arrays[0]) == np.bytes_:
        return [sum(ast.literal_eval(da.decode('utf-8'))) for da in diagonal_arrays]
    else:
        pdb.set_trace()
        return [sum(ast.literal_eval(da)) for da in diagonal_arrays]
    

def indices_by_key_value(h5, key_ref, idx, value):
    # collect all sessions with target session name
    idx_list=[]
    num_experiments = len(h5[key_ref])
    for experiment_idx in range(num_experiments):
        session_name = h5[key_ref][experiment_idx][idx]
        session_name = bytes_to_str(session_name)
        value = bytes_to_str(value)
        if value in session_name:
            idx_list.append(experiment_idx)    

    return idx_list
    

def collect_ratings(rearranged_dissim_matrix, rearranged_class_list, class_pair):
    
    """
    Collects dissim ratings relating to the given class pair.
    The order of the class pair is important to dismiss reordered version being collected
    from the similarity matrix.
    
    If there are 3 versions of each class, this would return 9 ratings
    If there were 3 of one class and 2 of another, only 6 ratings would be returned
    """

    # find element in dissimilarity matrix for given experiment entry
    ref_class, comp_class = class_pair
    class_pair_comparisons = []
#     pdb.set_trace()
    for class_idx_row, class_val_row in enumerate(rearranged_class_list):
        if class_val_row == ref_class:
#             print('class_val_row', class_val_row)
            matrix_row_val = class_idx_row
            for class_idx_col, class_val_col in enumerate(rearranged_class_list):
                if class_val_col == comp_class:
#                     print('class_val_col', class_val_col)
                    matrix_col_val = class_idx_col
                    comparison = rearranged_dissim_matrix[matrix_row_val,matrix_col_val]
                    class_pair_comparisons.append(comparison)

    return class_pair_comparisons



# these rating lists are grouped by pairwise comparisons (of which there are 15 conditions)
def class_rating_distributions(h5_ram, parent_list, class_pairs, id_string_list,
                               directory_name, show_plot=True, print_monitor=False, label_list=None, save_plot=False):
    
    """
    Get flattened list of all ratings for given list of experiment indices,
    (usually pertaining to a condition group).
    Optionally print these out.

    Return a list nested by the following structure:
        Class ratings -> condition group -> session ratings -> individual ratings
    
    """
    
    
    if label_list == None:
        label_list = ['straight', 'belt', 'breathy', 'fry', 'vibrato']
    all_class_distances_all_lists = []
    
    # iterate through class pair conditions
    for class_pair in class_pairs:
        specific_class_all_lists = []
        
        # iterate through each session in given parent list, and collect class_pair dissimilarity ratings
        for parent_idx, condition_list in enumerate(parent_list):
            condition_list_distances = []
            if print_monitor:
                print('condition_idx', parent_idx)
            
            # go through each experiemnt index
            for experiment_idx in condition_list:
                
                # get order of class names for this experiment entry
                rearranged_dissim_matrix = h5_ram['dissim_matrix'][experiment_idx]
                rearranged_ref_audio = h5_ram['rearrangedReferenceAudioNames'][experiment_idx]
                rearranged_class_list = strings_to_classes(rearranged_ref_audio, label_list)
                
                class_pair_comparisons = collect_ratings(rearranged_dissim_matrix,
                                                         rearranged_class_list, class_pair)
                if print_monitor:
                    print(experiment_idx, 'class_pair_comparisons:', class_pair_comparisons)
                
                # nest given dissim rating in parent list
                condition_list_distances.append(class_pair_comparisons)
                try:
                    dir_path = directory_name +'/' +id_string_list[parent_idx]
                    if not os.path.exists:
                        os.makedirs(dir_path, exist_ok=True)
                except:
                    pdb.set_trace()

            specific_class_all_lists.append(condition_list_distances)
            flattened_distances = [rating for cond_list in condition_list_distances for rating in cond_list]
            
            # PLOT AND SAVE HISTOGRAM
            if show_plot==True:
                plt.hist(flattened_distances, bins=10)
                title ='class {0} to {1} ratings'.format(class_pair[0], class_pair[1])
                # plt.title(title)
                plt.tight_layout()
                if save_plot:
                    plt.savefig(directory_name +'/' +id_string_list[parent_idx] +'/' +title)
                plt.show()
                plt.close()
            
        all_class_distances_all_lists.append(specific_class_all_lists)
    
    all_class_distances_all_lists = np.asarray(all_class_distances_all_lists)
    np.save(directory_name +'/' +directory_name, all_class_distances_all_lists)
    
    return all_class_distances_all_lists