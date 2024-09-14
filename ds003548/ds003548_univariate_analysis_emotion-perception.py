#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:36:20 2024

@author: pp262170
"""

#%%##### IMPORT
import os
import glob
import pandas as pd
import numpy as np
from nilearn import plotting, image
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from nilearn.reporting import get_clusters_table
from nilearn.glm.second_level import SecondLevelModel

#%%##### FUNCTION

def load_data(sub):
    """
    Load functional, onset, and confound files for a given subject.
    """
    data_path = os.path.join(derivatives, 'fmriprep-22.1.1', f'sub-{sub}')
    path_func = os.listdir(os.path.join(data_path, 'func'))
    func_file = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
    func_file.sort()
    
    path_onsets = os.listdir(os.path.join(main_path, dataset, 'rawdata'))
    onset_file = [item for item in path_onsets if item.endswith('events.tsv')]
    onset_file.sort()
    
    confound_file = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file.sort()
    
    return func_file, onset_file, confound_file, data_path

def load_event(sub, onset, func, confound, data_path, func_files, onset_list, confounds):
    """
    Load and preprocess event-related data for a given subject, including functional images and confounds.
    """
    data = pd.read_csv(os.path.join(main_path, dataset, 'rawdata', onset), sep='\t')
    
    onsets_tr_a = data[['trial_type', 'onset', 'duration']]
    onsets_tr = onsets_tr_a.copy()
    cond_names = ['neutral', 'sad', 'angry']
    
    conditions = onsets_tr[onsets_tr['trial_type'].isin(cond_names)]

    func_path = os.path.join(data_path, 'func')
    fmri_data = image.smooth_img(os.path.join(func_path, func), fwhm=8)  # Smoothing with FWHM of 8
    
    confound = pd.read_csv(os.path.join(func_path, confound), sep='\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]

    func_files.append(fmri_data)
    onset_list.append(conditions)
    confounds.append(confound)
    
    return func_files, onset_list, confounds, conditions

def first_level(func_files, onset_list, confounds):
    """
    Fit a first-level GLM model to the provided functional images and design matrices.
    """
    TR = 2
    
    glm = FirstLevelModel(
        t_r=TR,
        mask_img=mask_name,
        high_pass=0.008,
        smoothing_fwhm=4
    )
    
    first_level_model = glm.fit(func_files, events=onset_list, confounds=confounds)
    design_matrix = first_level_model.design_matrices_[0]
    
    return design_matrix

def make_localizer_contrasts(design_matrix):
    """
    Define contrasts for the localizer task based on the design matrix.
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }
    
    contrasts["LookNeg"] = contrasts["sad"] + contrasts["angry"]
    contrasts["LookNeu"] = contrasts["neutral"]
    
    contrasts = {
        "neg - neu": contrasts["LookNeg"] - contrasts["LookNeu"],
        "neg": contrasts["LookNeg"]
    }
    
    return contrasts

def plot_contrast(glm, contrast_val, contrast_id, nb_run):
    """
    Compute and return the contrast map for a given GLM model.
    """
    summary_statistics_run = glm.compute_contrast(
        contrast_val, output_type="all"
    )
    
    z_map = threshold_stats_img(summary_statistics_run['z_score'], 
                                alpha=1, height_control=None, cluster_threshold=0)
    
    return z_map

#%%##### PATH
main_path = '/neurospin/nfbd/Decoding/'
dataset = 'ds003548'
derivatives = os.path.join(main_path, dataset, 'derivatives/')

# List of masks to process
masks = [
    'frontal_mask.nii',
    'limbic_mask.nii',
    'fronto_limbic.nii',
    'wb_wfu.nii',
    'wb-wfu_without-frontal.nii',
    'wb-wfu_without-fronto-limbic.nii',
    'wb-wfu_without-limbic.nii'
]

for mask in masks:
    mask_name = os.path.join(main_path, 'code', 'mask', mask)
    
    sub_file = glob.glob(os.path.join(derivatives, 'fmriprep-22.1.1', 'sub-*'))
    sub_file = [item for item in sub_file if not item.endswith('.html')]
    
    sub_label_bmaps = []
    conditions_label_bmaps = []
    cmap = []
    
    for i in range(1, len(sub_file) + 1):
        if i == 2:  # Skip specific subject if needed
            continue
        
        sub = f'{i:02d}'
        print(f'Subject: {sub}')
        
        func_files = []
        onset_list = []
        confounds = []
        design_matrices = []
        
        func_file, onset_file, confound_file, data_path = load_data(sub)
        
        for k in range(1, len(onset_file) + 1): 
            run = f'{k}'
            print(f'Run: {run}')
            
            onset = onset_file[int(run) - 1]
            func = func_file[int(run) - 1]
            confound = confound_file[int(run) - 1]
    
            func_files, onset_list, confounds, conditions = load_event(
                sub, onset, func, confound, data_path, func_files, onset_list, confounds
            )
            design_matrix = first_level(func_files, onset_list, confounds)
            design_matrices.append(design_matrix)
    
        cut_coords = [-27, -4, -20]  # Amygdala coordinates

        print('Step: Save beta-maps')
        glm_run = FirstLevelModel(
            t_r=2, 
            high_pass=0.008,
            smoothing_fwhm=None
        )
        
        fmriglm_multirun = glm_run.fit(func_files, design_matrices=design_matrices)   
        contrast_val = [np.array([[1, -2, 1]])] * len(design_matrices)
        
        zmap = fmriglm_multirun.compute_contrast(contrast_val, output_type="z_score")
        
        cmap.append(zmap)
        conditions_label_bmaps.append('neg - neu')
        sub_label_bmaps.append(sub)
    
    #%% SECOND LEVEL ANALYSIS
    n_samples = len(cmap)
    design_matrix = pd.DataFrame([1] * n_samples, columns=["intercept"])

    second_level_model = SecondLevelModel(n_jobs=2).fit(
        cmap, design_matrix=design_matrix
    )

    z_map = second_level_model.compute_contrast(output_type="z_score")

    # Plot the unthresholded map
    display = plotting.plot_stat_map(z_map, title="Raw z map")

    # Threshold map at alpha=0.05
    thresholded_map2, threshold2 = threshold_stats_img(
        z_map,
        alpha=0.05,
        height_control="fpr",
        cluster_threshold=10,
        two_sided=True
    )

    # Plot the thresholded map
    plotting.plot_stat_map(
        thresholded_map2,
        cut_coords=display.cut_coords,
        title="Thresholded z map, expected fdr = .05",
        threshold=threshold2
    )

    # Save the map
    thresholded_map2.to_filename(os.path.join(
        derivatives, 'nilearn', 'second_level_analysis', 'contrast_img',
        f'{mask.split(".")[0]}_thresholdedmap.nii'
    ))

    # Save cluster table
    table = get_clusters_table(
        thresholded_map2, stat_threshold=threshold2, cluster_threshold=10
    )
    table.to_csv(os.path.join(
        derivatives, 'nilearn', 'second_level_analysis', 'contrast_img',
        f'{mask.split(".")[0]}_negneu_table.csv'
    ))


