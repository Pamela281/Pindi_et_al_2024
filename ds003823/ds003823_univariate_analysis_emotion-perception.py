#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 15:20:53 2024

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


#%%##### FUNCTIONS

def outliers(fd_lim: int):
    """
    Load the list of subjects with outlier framewise displacement values.
    """
    sub_outliers = pd.read_csv(os.path.join(main_path, dataset, 'derivatives', 'mriqc', f'name_suboutliers_mriqc_fd-lim-{fd_lim}.csv'), sep=',')
    return sub_outliers['participant_id'].values.tolist()

def load_data(sub):
    """
    Load functional, onset, and confound files for a given subject.
    """
    data_path = os.path.join(derivatives, 'fmriprep-22.1.1', sub)
    path_func = os.listdir(os.path.join(data_path, 'ses-pre', 'func'))
    func_file = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
    func_file.sort()
        
    path_onsets = os.listdir(os.path.join(main_path, dataset, 'rawdata', sub, 'ses-pre', 'func'))
    onset_file = [item for item in path_onsets if item.endswith('emotionRegulation_events.tsv')]
    onset_file.sort()
    
    confound_file = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file.sort()
    
    return func_file, onset_file, confound_file, data_path

def load_event(sub, data_path, func_files, onset_list, confounds):
    """
    Load event data, functional data, and confound data for a given run.
    """
    event_file = os.path.join(main_path, dataset, 'rawdata', sub, 'ses-pre', 'func', f'{sub}_ses-pre_task-emotionRegulation_events.tsv')
    data = pd.read_csv(event_file, sep='\t')
    
    # Keep only columns trial_type, onset and duration
    onsets_tr = data[['trial_type', 'onset', 'duration']]   
    conditions = onsets_tr[onsets_tr['trial_type'].isin(['vn', 'vt'])]  # Filter conditions of interest

    # Load functional data
    func = os.path.join(data_path, 'ses-pre', 'func')
    fmri_data = image.smooth_img(os.path.join(func, func_files[0]), fwhm=8)
    
    # Load confounds
    confound = pd.read_csv(os.path.join(func, confound_file[0]), sep='\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]

    func_files.append(fmri_data)
    onset_list.append(conditions)
    confounds.append(confound)
    
    return func_files, onset_list, confounds       

def first_level(func_files, onset_list, confounds):
    """
    Perform first-level GLM analysis on the functional data.
    """
    TR = 2.4
    
    glm = FirstLevelModel(
        t_r=TR,
        mask_img=mask_name,
        high_pass=0.008,
        smoothing_fwhm=4
    )
        
    first_level_model = glm.fit(func_files, events=onset_list, confounds=confounds)
    design_matrices = first_level_model.design_matrices_
    
    return design_matrices, glm

def make_localizer_contrasts(design_matrix):
    """
    Create contrasts for the localizer task.
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }
    
    contrasts["LookNeg"] = contrasts["vn"]
    contrasts["LookNeu"] = contrasts["vt"]
    
    contrasts = {
        "neg - neu": contrasts["LookNeg"] - contrasts["LookNeu"],
        "neu - neg": contrasts["LookNeu"] - contrasts["LookNeg"],
        "neu": contrasts["LookNeu"],
        "neg": contrasts["LookNeg"]
    }
    
    return contrasts
                        
def plot_contrast(glm, contrast_val, contrast_id, nb_run):
    """
    Compute and plot the contrast map for a given GLM model.
    """
    summary_statistics_run = glm.compute_contrast(contrast_val, output_type="all")
    thresholded_map, threshold = threshold_stats_img(summary_statistics_run['z_score'], alpha=1, height_control='fpr', cluster_threshold=0)
    
    # Plot the thresholded map
    plotting.plot_stat_map(thresholded_map, cut_coords=cut_coords)
    
    return thresholded_map

#%%##### PATH SETUP
main_path = '/neurospin/nfbd/Decoding/'
dataset = 'ds003823'
derivatives = os.path.join(main_path, dataset, 'derivatives/')
sub_file = [item for item in os.listdir(os.path.join(derivatives, 'fmriprep-22.1.1/')) if not item.endswith('.html') and item.startswith('sub-')]

# List of masks to process
masks = ['frontal_mask.nii', 'limbic_mask.nii', 'fronto_limbic.nii', 'wb_wfu.nii']


for mask in masks:
    mask_name = os.path.join(main_path, 'code', 'mask', mask)
    
    sub_label_bmaps = []
    conditions_label_bmaps = []
    cmap = []

    for sub in sub_file:
        sub_outliers = outliers(0.3)
        
        if sub in sub_outliers or sub == 'sub-7007':
            continue
        
        print(f'Subject: {sub}')
        
        func_files = []
        onset_list = []
        confounds = []
    
        func_file, onset_file, confound_file, data_path = load_data(sub)
        func_files, onset_list, confounds = load_event(sub, data_path, func_files, onset_list, confounds)
            
        design_matrices, glm = first_level(func_files, onset_list, confounds)
    
        cut_coords = [-27, -4, -20]  # Coordinates for amygdala

        print('Step: save beta-maps')
        fmriglm_multirun = glm.fit(func_files, design_matrices=design_matrices)   
        
        contrast_val = [np.array([[1, -2, 1]])] * len(design_matrices)
        zmap = fmriglm_multirun.compute_contrast(contrast_val, output_type="all")
        
        cmap.append(zmap)
        conditions_label_bmaps.append('neg - neu')
        sub_label_bmaps.append(sub)

    #%% SECOND LEVEL ANALYSIS
    n_samples = len(cmap)
    design_matrix = pd.DataFrame([1] * n_samples, columns=["intercept"])

    second_level_model = SecondLevelModel(n_jobs=2).fit(cmap, design_matrix=design_matrix)
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

    # Save the thresholded map
    thresholded_map2.to_filename(
        os.path.join(main_path, dataset, 'derivatives', 'nilearn', 'second_level_analysis', 'contrast_img', 
                     f'{mask_name.rsplit("/", 8)[-1].rsplit(".")[0]}_thresholdedmap.nii')
    )

    # Generate and save cluster table
    table = get_clusters_table(thresholded_map2, stat_threshold=threshold2, cluster_threshold=10)
    table.to_csv(
        os.path.join(main_path, dataset, 'derivatives', 'nilearn', 'second_level_analysis', 'contrast_img',
                     f'{mask_name.rsplit("/", 8)[-1].rsplit(".")[0]}_negneu_table.csv')
    )

