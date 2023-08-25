import itertools
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pdb
import warnings

from cliffs_delta import cliffs_delta
from scipy.stats import chi2, rankdata, pearsonr, kendalltau, spearmanr, norm, skew, shapiro, kstest, linregress

from utils import diagonal_array_handling



def show_hist(sample, bins=15, title=''):
    print(f'Size of sample is {len(sample)}')
    plt.hist(np.asarray(sample), bins=bins)
    # plt.title(title)
    plt.show()
    plt.close()


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
        raise Exception('write some alternative code here for if your samples are too small!')
    return h_value, p_value


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
    
    ordinal_cohen_d = z_value/math.sqrt(sample_a_size*sample_b_size) # uncertain if this is the right way

#    print(sample_a_mean, sample_b_mean)
    return sample_a_mean, sample_b_mean, abs(z_value), ordinal_cohen_d, u_value, sample_a_size, sample_b_size


from scipy.stats import mannwhitneyu

def custom_mann_whit_better(sample_list, sample_names, print_all, sig_thresh):

    stat, pval = mannwhitneyu(sample_list[0], sample_list[1])
    d, res = cliffs_delta(sample_list[0], sample_list[1])
    if pval < sig_thresh:
        result = stat, pval, f'{sample_names[0]} median: {np.median(sample_list[0])}', f'{sample_names[1]} median: {np.median(sample_list[1])}', f'effect: {d}' 
        if print_all:
            print(result)
    else:
        if print_all:
            print('no diff between samples', sample_names[0], sample_names[1])
        result = None
    return result
        

def nonpara_multisample_stat_test(samples_list, sig_thresh, sample_names, print_all=True):
    
    z_value_threshold = norm.ppf(1 - sig_thresh)
    num_conditions = len(samples_list)
    d_of_f = num_conditions-1
    mann_whit_results = []

    if num_conditions>2:
#         h_value, p_value = custom_kruskal(samples_list)
        p_value = 0.01
        if p_value < sig_thresh:
    
            condition_pairs = list(itertools.combinations(range(num_conditions), 2))
            # adjust sig levels
            bonferri_corrected_sig_thresh = sig_thresh/d_of_f
            bonferri_corrected_z_threshold = norm.ppf(1 - bonferri_corrected_sig_thresh)
            # go through every combination of conditions
            for con_pair_idx, (condition_a, condition_b) in enumerate(condition_pairs):
                contrast_conditions = (samples_list[condition_a], samples_list[condition_b])
                contrast_names = (sample_names[condition_a], sample_names[condition_b])
#                 custom_result = custom_mann_whit(contrast_conditions)
                man_whit_result = custom_mann_whit_better(contrast_conditions, contrast_names, print_all, sig_thresh)
                mann_whit_results.append(man_whit_result)
        else:
            if print_all:
                print('No signifncant diffs using Kruskal Wallis')
                
    else:
        assert num_conditions == 2
        
        # FIXME: code smell!
        condition_pairs = list(itertools.combinations(range(num_conditions), 2))
        # adjust sig levels
        bonferri_corrected_sig_thresh = sig_thresh/d_of_f
        bonferri_corrected_z_threshold = norm.ppf(1 - bonferri_corrected_sig_thresh)
        # go through every combination of conditions
        for con_pair_idx, (condition_a, condition_b) in enumerate(condition_pairs):
            contrast_conditions = (samples_list[condition_a], samples_list[condition_b])
            contrast_names = (sample_names[condition_a], sample_names[condition_b])
#                 custom_result = custom_mann_whit(contrast_conditions)
            man_whit_result = custom_mann_whit_better(contrast_conditions, contrast_names, print_all, sig_thresh)
            mann_whit_results.append(man_whit_result)
    
    return mann_whit_results


def normality_stat_test(data, test_type='shaprio', refuse_minimal=20, show_plot=True):

    if type(data[0]) == bytes or type(data[0]) == np.bytes_:
        data = [float(b.decode('utf-8')) for b in data]
    if len(data) < refuse_minimal:
        warnings.warn('Data sample size is less than minimal value {refuse_minimal}.')
    
    if show_plot:
        show_hist(data, bins=15, title='')
    skewness = skew(data)
    if test_type == 'shaprio':
        stat, p = shapiro(data) # SHAPIRO-WILK - if p value less than the accepted 0.05, data is NOT normal
    elif test_type == 'kstest':
        stat, p = kstest(data, 'norm')

    return stat, p, skewness


def normality_test_wrapper(scores_by_condition, condition_names, variable_name, refuse_minimal=20, show_plot=True):
    for cond_idx, cond_group_scores in enumerate(scores_by_condition):
        stat, p, skewness = normality_stat_test(cond_group_scores, refuse_minimal=refuse_minimal, show_plot=show_plot)
        try:
            if p > 0.05:
                print('variable:', variable_name, ', condition name:', condition_names[cond_idx],', - NORMAL DISTRIBUTION')
            else:
                print('variable:', variable_name, ', condition name:', condition_names[cond_idx],', - NOT NORMAL DISTRIBUTION')
        except:
            pdb.set_trace()


def array_column_correlations(array, feat_names, profile_label_dict, sig_thresh, test_type, show_plot=True, save_plot=True):
    """
    Finds correlations between different features provided by participants,
    which include their questionnaire information and rating clustering metrics.
    
    """
    
    def get_array_feats(idx, array, feat_names):
        """Extract features from array at index, while allowing for specific
        processing if the index calls on the identity recognition features"""

        feats_name = feat_names[idx]
        feats = array[:,idx]
        if idx == 5:
            feats = diagonal_array_handling(feats)
        
        feats = np.asarray(feats).astype(float)
            
        return feats_name, feats
    
    def regression_line(x):
        return slope * x + intercept
    
    
    # setup variables
    num_feats = array.shape[1]
    correlation_dict = {}
    correlation_list = []
    
    # tests every collection of values against every other collection of values for correlation
    for i in range(0, num_feats):

        for j in range(0, num_feats):
            
            if i == j: continue
            
            i_feats_name, i_feats = get_array_feats(i, array, feat_names)
            j_feats_name, j_feats = get_array_feats(j, array, feat_names)
                
            if test_type=='pearsonr':
                correlation_coefficient, p_val = pearsonr(i_feats, j_feats)
            elif test_type=='kendalltau':
                correlation_coefficient, p_val = kendalltau(i_feats, j_feats)
            elif test_type=='spearmanr':
                correlation_coefficient, p_val = spearmanr(i_feats, j_feats)
#             print(f'corr: {i_feats_name} x {j_feats_name}:', correlation_coefficient, p_val)
                
            # add result to list if p_val is low enough and conditions haven't been compared
            if p_val <= sig_thresh:
                
                # check to see if this feature comparison was already recorded in dict
                if f'{i_feats_name} x {j_feats_name}' in correlation_dict.keys():
                    continue
                elif f'{j_feats_name} x {i_feats_name}' in correlation_dict.keys():
                    continue
                    
                else:
                    entry = (i_feats_name.split(' ')[0], j_feats_name.split(' ')[0], round(float(correlation_coefficient), 2), float(p_val))
                    correlation_dict[f'{i_feats_name} x {j_feats_name}'] = entry
                    correlation_list.append(entry)
                    print(entry)
                    
                    if show_plot:
                        y_min = np.min(j_feats)
                        y_max = np.max(j_feats)
                        y_interval = round((y_max-y_min)/10, 2)  # Set the desired interval
                        x_min = np.min(i_feats)
                        x_max = np.max(i_feats)
                        num_cats = len(np.unique(i_feats))
                        
                        if num_cats<5:
                            print('using box plot')
                            subsubdir = 'corr_box'
                            # must be the ordinal data, so make box plots
                            figure = plt.figure(figsize =(20, 10))  
                            
                            vals_by_cat = [[] for i in range(num_cats)]
                            for i in range(num_cats):
                                vals_by_cat[i] = j_feats[np.where(i_feats == i)[0]]
                                plt.boxplot(vals_by_cat)
                                labels_tuple = profile_label_dict[i_feats_name][1]
                                plt.xticks(range(1, len(labels_tuple) + 1), labels_tuple)
                                plt.yticks(fontsize=18)
                                plt.xticks(fontsize=18)
                                plt.xlabel(i_feats_name, fontsize=18)
                                plt.ylabel(j_feats_name, fontsize=18)
                        else:
                            subsubdir = 'corr_scat'
                            slope, intercept, r_value, p_value, std_err = linregress(i_feats, j_feats)
                            
                            x_interval = round((x_max-x_min)/10, 2)
                            
                            if 'MSI' in i_feats_name:
                                plt.xticks(np.arange(49,96,5))
                            else:
                                plt.xticks(np.linspace(0,1,11))
                                
                            if 'MSI' in j_feats_name:
                                plt.yticks(np.arange(49,96,5))
                            else:
                                plt.yticks(np.linspace(0,1,11))
                            
                            plt.scatter(i_feats, j_feats, label=f'Correlation Coefficient: {correlation_coefficient:.2f}')
                            plt.plot(i_feats, regression_line(i_feats), color='red', label=f'Regression Line (R-squared = {r_value**2:.2f})')
                            plt.ylabel(j_feats_name)
                            plt.xlabel(i_feats_name)
                            plt.legend()
                        
                        title = f'{test_type}_correlation'
#                         # plt.title(title)
                        
                        if save_plot:
                            subdir = 'correlations'
                            if not os.path.exists(subdir):
                                os.mkdir(subdir)
                            plt.savefig(os.path.join(subdir, subsubdir, f'{j_feats_name} x {i_feats_name}_' +title))
                            
                        plt.show()
                        plt.close()
                                        
    return correlation_dict, correlation_list