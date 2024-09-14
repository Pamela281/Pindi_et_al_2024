#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:27:54 2024

@author: pp262170
"""

#%%#########################################################################
# IMPORT -----------------------------------------------------------------
############################################################################
import os
import glob
import pandas as pd
import numpy as np
from nilearn.maskers import NiftiMasker
from nilearn import plotting, image
from nilearn.image import load_img, index_img, concat_imgs, high_variance_confounds, new_img_like
from nilearn.plotting import view_img, plot_roi, plot_stat_map, show, plot_design_matrix
from nilearn.decoding import FREMClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold, ShuffleSplit, StratifiedShuffleSplit, permutation_test_score
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level import FirstLevelModel
from sklearn.svm import SVC
from sklearn.utils import shuffle
from nilearn.reporting import get_clusters_table

#%%#########################################################################
# FUNCTIONS --------------------------------------------------------------
############################################################################

def load_data(sub):
    """
    Load functional, onset, and confound files for a given subject.
    """
    data_path = os.path.join(derivatives, f'fmriprep-22.1.1/sub-{sub}/')
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
    
    # Load and smooth functional data
    func_path = os.path.join(data_path, 'func')
    fmri_data = image.smooth_img(os.path.join(func_path, func), fwhm=6)
    
    # Load confounds
    confound = pd.read_csv(os.path.join(func_path, confound), sep='\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]
    
    func_files.append(fmri_data)
    onset_list.append(conditions)
    confounds.append(confound)

    return func_files, onset_list, confounds, conditions   

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
    
    # Define more relevant contrasts
    contrasts = {
        "neg - neu": contrasts["LookNeg"] - contrasts["LookNeu"],
        "neu - neg": contrasts["LookNeu"] - contrasts["LookNeg"],
        "neu": contrasts["LookNeu"],
        "neg": contrasts["LookNeg"]
    }
    
    return contrasts

def first_level(func_files, onset_list, confounds):
    """
    Fit a first-level GLM model to the provided functional images and design matrices.
    """
    TR = 2
    
    glm = FirstLevelModel(
        t_r=TR,
        mask_img=mask_name,
        high_pass=1/128,
        smoothing_fwhm=None
    )
    
    first_level_model = glm.fit(func_files, events=onset_list, confounds=confounds)
    design_matrices = first_level_model.design_matrices_
    
    return design_matrices, glm
                       
def plot_contrast(glm, contrast_val, contrast_id):
    """
    Compute and return the contrast map for a given GLM model.
    """
    summary_statistics_run = glm.compute_contrast(
        contrast_val, output_type="all"
    )
    
    return summary_statistics_run

def decodeur(contrast_imgs, conditions_label, groups, mask_name, cv):
    """
    Train a FREMClassifier using the provided contrast images and labels.
    """
    decoder_FREM = FREMClassifier(
        estimator='svc',
        mask=mask_name,
        cv=cv,
        clustering_percentile=10,
        screening_percentile=20,
        scoring='roc_auc',
        smoothing_fwhm=4,
        standardize=False,
        verbose=3
    )
    
    decoder_FREM.fit(contrast_imgs, conditions_label, groups=groups)
    return decoder_FREM

def permutation_test(X, y, model, groups, n_permutations=1000):
    """
    Perform permutation test to evaluate the significance of the model's performance.
    """
    model.fit(X, y, groups=groups)
    y_pred = model.predict(X)
    actual_score = np.mean(list(model.cv_scores_['neg']))
    permuted_scores = []
    
    for _ in range(n_permutations):
        y_permuted = shuffle(y)
        model.fit(X, y_permuted, groups=groups)
        y_permuted_pred = model.predict(X)
        permuted_score = accuracy_score(y, y_permuted_pred)
        permuted_scores.append(permuted_score)
    
    p_value = (np.sum(np.array(permuted_scores) >= actual_score) + 1) / (n_permutations + 1)
    return actual_score, permuted_scores, p_value

#%%#########################################################################
# MAIN SCRIPT ------------------------------------------------------------
############################################################################

main_path = '/neurospin/nfbd/Decoding/'
dataset = 'ds003548'
derivatives = os.path.join(main_path, dataset, 'derivatives/')

# List of masks to process
masks = ['frontal_mask.nii', 'limbic_mask.nii', 'fronto_limbic.nii', 'wb_wfu.nii', 
         'wb-wfu_without-frontal.nii', 'wb-wfu_without-fronto-limbic.nii', 'wb-wfu_without-limbic.nii']

# Loop over each mask
for mask in masks:
    mask_name = os.path.join(main_path, 'code', 'mask', mask)
    
    sub_file = glob.glob(os.path.join(derivatives, 'fmriprep-22.1.1', 'sub-*'))
    sub_file = [item for item in sub_file if not item.endswith('.html')]
    
    sub_label_bmaps, conditions_label_bmaps, contrast_imgs = [], [], []

    for i in range(1, len(sub_file) + 1):
        if i == 2:  # Skip specific subject if needed
            continue
        
        sub = f'{i:02d}'
        print(f'Subject: {sub}')
        
        func_files, onset_list, confounds = [], [], []
        
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

        design_matrices, glm = first_level(func_files, onset_list, confounds)

        print('Step: Save beta-maps')
        for i in range(len(design_matrices)):
            contrasts = make_localizer_contrasts(design_matrices[i])
            fmri_glm = glm.fit(func_files[i], design_matrices=design_matrices[i])
            
            for contrast_id, contrast_val in contrasts.items():
                if contrast_id in ['neg', 'neu']:
                    summary_statistics_run = plot_contrast(fmri_glm, contrast_val, contrast_id)
                    contrast_imgs.append(summary_statistics_run['z_score'])
                    conditions_label_bmaps.append(contrast_id)
                    sub_label_bmaps.append(sub)
    
    # Run decoder
    print('Running decoder')
    decoder_FREM = decodeur(contrast_imgs, conditions_label_bmaps, sub_label_bmaps, mask_name, StratifiedShuffleSplit(n_splits=5, random_state=42))
    weight_img = decoder_FREM.coef_img_['neg']
    weight_img.to_filename(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'weights_img', f'{mask.split(".")[0]}_neg_weights.nii'))

    # Permutation test for thresholding
    print('Running permutation test')
    decoder_FREM_permut = FREMClassifier(
        estimator='svc',
        mask=mask_name,
        cv=StratifiedShuffleSplit(n_splits=5, random_state=42),
        clustering_percentile=10,
        screening_percentile=20,
        scoring='accuracy',
        smoothing_fwhm=4,
        standardize=True,
        verbose=3
    )

    decoder_FREM_permut.fit(contrast_imgs, conditions_label_bmaps, groups=sub_label_bmaps)
    coef_img = decoder_FREM.coef_img_['neg']
    coef_data = coef_img.get_fdata()

    n_permutations = 1000
    null_distributions = []

    for i in range(n_permutations):
        print(f'Permutation {i + 1}/{n_permutations}')
        y_permuted = shuffle(conditions_label_bmaps)
        decoder_FREM_permut.fit(contrast_imgs, y_permuted, groups=sub_label_bmaps) 
        permuted_score = decoder_FREM_permut.score(contrast_imgs, y_permuted)
        null_distributions.append(permuted_score)

    threshold = np.percentile(null_distributions, 95)
    thresholded_data = np.where(np.abs(coef_img.get_fdata()) > 1 - threshold, coef_img.get_fdata(), 0)
    thresholded_img = new_img_like(coef_img, thresholded_data)

    # Save thresholded image and clusters table
    thresholded_img.to_filename(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'weights_img', f'{mask.split(".")[0]}_neg_weights_thr.nii'))
    clusters_tb = get_clusters_table(thresholded_img, 1 - threshold)
    clusters_df = pd.DataFrame(clusters_tb)
    clusters_df.to_csv(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'csvfiles_ds003548', f'{mask.split(".")[0]}_neg_clusters_table.csv'), index=False)

    # Perform permutation test
    print('Performing permutation test')
    actual_score, permuted_scores, pvalue = permutation_test(contrast_imgs, conditions_label_bmaps, decoder_FREM_permut, sub_label_bmaps, n_permutations=1000)

    # Save summary and permutation scores
    summary = {
        'cv_scores': list(decoder_FREM.cv_scores_['neg']),
        'mask': mask.split('.')[0]
    }
    df2 = pd.DataFrame.from_records(summary)
    df2.to_csv(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'csvfiles_ds003548', f'summary_table_data_across_sub_{mask.split(".")[0]}_noerror.csv'))

    # Opening a file named "permutation_score.txt" in write mode
    file = open(derivatives+'nilearn/first_level_analysis/csvfiles_ds003548/permutation_score_' + 
               mask_name.rsplit('/', 8)[-1].rsplit('.')[0]+'.txt', "w")

    lines = ['Permutation score = ' + str(np.mean(np.array(permuted_scores))), "P-value_score = " + str(pvalue), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "\nContrast =" + list(contrasts)[0]]

    with file as f:
        for line in lines:
            f.write(line)
            f.write('\n')

