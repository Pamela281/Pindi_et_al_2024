#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:14:07 2024

@author: pp262170
"""

#%%##### IMPORT
import os, glob
import pandas as pd
import numpy as np
from nilearn import plotting, image
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import SecondLevelModel

#%%##### FUNCTION
# Define helper functions for data processing and analysis

def outliers(fd_lim: int):
    """
    Identify outlier subjects based on framewise displacement limit.
    
    Args:
    fd_lim (int): Framewise displacement limit
    
    Returns:
    list: List of outlier subject names
    """
    sub_outliers = pd.read_csv(derivatives + '/mriqc-22.0.6/name_suboutliers_mriqc_fd-lim-' + str(fd_lim) + ".csv", sep=',')
    sub_outliers = sub_outliers['bids_name'].values.tolist()
    
    return sub_outliers

def load_data(sub):
    """
    Load functional MRI data, onset files, and confound files for a given subject.
    
    Args:
    sub (str): Subject ID
    
    Returns:
    tuple: Lists of functional files, onset files, confound files, and data path
    """
    data_path = derivatives + '/fmriprep-22.1.1/sub-' + sub + '/'
    path_func = os.listdir(data_path + '/func/')
    func_file = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
    func_file.sort()
        
    path_onsets = os.listdir("/neurospin/nfbd/Decoding/ds000108/rawdata/sub-" + sub + "/func/")
    onset_file = [item for item in path_onsets if item.endswith('events.tsv')]
    onset_file.sort()
    
    confound_file = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file.sort()
    
    return func_file, onset_file, confound_file, data_path

def load_event(sub, onset, func, confound, data_path, func_files, onset_list, confounds):
    """
    Process event data for a single run, including smoothing fMRI data and preparing confounds.
    
    Args:
    sub (str): Subject ID
    onset (str): Onset file name
    func (str): Functional MRI file name
    confound (str): Confound file name
    data_path (str): Path to data
    func_files (list): List to append processed fMRI data
    onset_list (list): List to append processed onset data
    confounds (list): List to append processed confound data
    
    Returns:
    tuple: Updated func_files, onset_list, confounds, and conditions
    """
    
    data = pd.read_csv(main_path + dataset + '/rawdata/sub-' + sub + "/func/" + onset, sep='\t')
    
    onsets_tr_a = data[['trial_type', 'onset']]
    onsets_tr = onsets_tr_a.copy()
    duration = np.hstack((onsets_tr.onset.values[1:] - onsets_tr.onset.values[:-1], 5))
    # Note the 5 at the end is arbitrary (fixme)
    onsets_tr['duration'] = duration
    cond_names = ['Reapp_Neg_Stim','Look_Neg_Stim']
    conditions = onsets_tr[onsets_tr['trial_type'].isin(cond_names)]

    func_path = data_path + 'func/'
    fmri_data = image.smooth_img(func_path + func, fwhm=8)
    
    # Load confounds
    confound = pd.read_csv(func_path + confound, sep='\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]
        
    func_files.append(fmri_data)
    onset_list.append(conditions)
    confounds.append(confound)
    
    # conditions = conditions.trial_type.unique() 
    
    return func_files, onset_list, confounds, conditions

def first_level(func_files, onset_list, confounds):
    """
    Perform first-level GLM analysis.
    
    Args:
    func_files (list): List of functional MRI data
    onset_list (list): List of onset data
    confounds (list): List of confound data
    
    Returns:
    tuple: Design matrices and GLM object
    """
    TR = 2
    
    glm = FirstLevelModel(
         t_r=TR,
         mask_img=mask_name,
         high_pass=1/128,
         smoothing_fwhm=None)
        # memory="nilearn_cache")
        
    first_level_model = glm.fit(func_files, events=onset_list, confounds=confounds) #add confounds
    
    # design_matrix = first_level_model.design_matrices_[0]
    design_matrices = first_level_model.design_matrices_
    
    return design_matrices, glm

def make_localizer_contrasts(design_matrix):
    """
    Create contrast matrices for the localizer task.
    
    Args:
    design_matrix (DataFrame): Design matrix from first-level analysis
    
    Returns:
    dict: Dictionary of contrast matrices
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
           column: contrast_matrix[i]
           for i, column in enumerate(design_matrix.columns)
       }
        
    contrasts["LookNeg"] = (
            contrasts["Look_Neg_Stim"])
    
    contrasts["ReappNeg"] = (
            contrasts["Reapp_Neg_Stim"])
    
    # Short dictionary of more relevant contrasts
    contrasts = {
        "reg - neg": contrasts["ReappNeg"] - contrasts["LookNeg"],
        # "neu - neg": contrasts["LookNeu"] - contrasts["LookNeg"],
        "reg" : contrasts['ReappNeg'],
        "neg" : contrasts['LookNeg']}
    
    return contrasts
                        
def plot_contrast(glm, contrast_val, contrast_id, nb_run):
    """
    Compute and plot contrast maps.
    
    Args:
    glm (FirstLevelModel): Fitted GLM object
    contrast_val: Contrast values
    contrast_id (str): Contrast identifier
    nb_run (int): Number of runs
    
    Returns:
    NiftiImage: Thresholded statistical map
    """

    # Compute the per-contrast z-map
    summary_statistics_run = glm.compute_contrast(
        contrast_val, output_type="all"
    )

    # Define the threshold
    thresholded_map, threshold = threshold_stats_img(summary_statistics_run['z_score'], 
                                alpha=1, height_control='fpr', cluster_threshold=0)
    
    # # Plot the thresholded map
    # plotting.plot_stat_map(
    # thresholded_map, cut_coords = cut_coords,
    # title=f'{contrast_id}, run {nb_run+1}, {mask_name.rsplit("/", 8)[-1].rsplit(".")[0]} , alpha = 0.05')

    # Save the figure in png format
    # plt.savefig(f'/neurospin/nfbd/Decoding/ds000108/derivatives/nilearn/first_level_analysis/lookNeu_vs_lookNeg_run{nb_run +1}.png')
    
    return thresholded_map

#%%##### PATH
# Set up paths and parameters
main_path = '/neurospin/nfbd/Decoding/'
dataset = 'ds000108'
derivatives = main_path + dataset + '/derivatives/'

# Get list of subject directories
sub_file = glob.glob(derivatives + 'fmriprep-22.1.1/sub-*')
sub_file = [item for item in sub_file if not item.endswith('.html')]

# Define masks for analysis
masks = ['frontal_mask.nii', 'limbic_mask.nii', 'fronto_limbic.nii', 'wb_wfu.nii']

# Iterate through each mask
for mask in masks:
    mask_name = main_path + 'code/mask/' + mask
    cmap = []

    # Process each subject
    for i in range(1, len(sub_file) + 1):
        # Skip certain subject numbers (likely due to data issues)
        if i in [6, 11, 25, 32]:
            i += 1
        if i == 12:
            i += 1
        
        sub = '%02d' % i
        print('Subject:' + sub)

        # Initialize lists for storing data
        func_files = []
        onset_list = []
        confounds = []

        # Load data for the current subject
        func_file, onset_file, confound_file, data_path = load_data(sub)

        # Process each run for the current subject
        for k in range(1, len(onset_file) + 1):
            run = '%02d' % k
            print('Run:' + run)

            # Load data for the current run
            onset = onset_file[int(run) - 1]
            func = func_file[int(run) - 1]
            confound = confound_file[int(run) - 1]

            # Check for outliers
            func_split = '_'.join(func.split('_space-MNI152NLin2009cAsym_desc-preproc_'))
            sub_outliers = outliers(0.3)
            if func_split.split('.nii.gz')[0] in sub_outliers:
                pass
            else:
                print('ok')
                func_files, onset_list, confounds, conditions = load_event(sub, onset, func, confound, data_path, func_files, onset_list, confounds)

        # Perform first-level analysis
        design_matrices, glm = first_level(func_files, onset_list, confounds)

        print('Step: save beta-maps')
        for i in range(len(design_matrices)):
        
            # Call the contrast specification within the function    
            contrasts = make_localizer_contrasts(design_matrices[i])
            fmri_glm = glm.fit(func_files[i], design_matrices=design_matrices[i])
            
            for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
                
                if contrast_id == 'reg - neg':
                    
                    thresholded_map = plot_contrast(fmri_glm, contrast_val, contrast_id, i)
                    
                    cmap.append(thresholded_map)

    #%% SECOND LEVEL ANALYSIS
    # Perform second-level analysis across all subjects
    n_samples = len(cmap)
    design_matrix = pd.DataFrame([1] * n_samples, columns=["intercept"])
    
    # Fit second-level model
    second_level_model = SecondLevelModel(n_jobs=2).fit(
        cmap, design_matrix=design_matrix)
    
    # Compute group-level z-map
    z_map = second_level_model.compute_contrast(output_type="z_score")
    
    # Plot the unthresholded map
    display = plotting.plot_stat_map(z_map, title="Raw z map")
    
    # Threshold map at alpha = 0.05
    thresholded_map2, threshold2 = threshold_stats_img(
        z_map,
        alpha=0.05,
        height_control="fpr",
        cluster_threshold=10,
        two_sided=True,
    )
    
    # Plot the thresholded map
    plotting.plot_stat_map(
        thresholded_map2,
        cut_coords=display.cut_coords,
        title="Thresholded z map, expected FDR = .05",
        threshold=threshold2,
    )
    
    # Save thresholded map
    thresholded_map2.to_filename('/neurospin/nfbd/Decoding/' + dataset + '/derivatives/nilearn/second_level_analysis/contrast_img/' +
                                 mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + '_reg_thresholdedmap.nii')
    
    # Generate and save cluster table
    table = get_clusters_table(
        thresholded_map2, stat_threshold=threshold2, cluster_threshold=10)
    
    table.to_csv('/neurospin/nfbd/Decoding/' + dataset + '/derivatives/nilearn/second_level_analysis/contrast_img/' +
                 mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + '_regneg_table.csv')
    

