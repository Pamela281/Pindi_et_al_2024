#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:49:27 2023

@author: pp262170
"""

#%% IMPORT -----------------------------------------------------------------
import os
import glob
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.image import new_img_like
from nilearn.decoding import FREMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from nilearn.glm.first_level import FirstLevelModel
from sklearn.utils import shuffle
from nilearn.reporting import get_clusters_table


#%% FUNCTION ------------------------------------------------------------

def outliers(fd_lim: int):
    """
    Load the list of subjects with outlier framewise displacement values.
    """
    sub_outliers = pd.read_csv(
        derivatives + '/mriqc-22.0.6/name_suboutliers_mriqc_fd-lim-' + str(fd_lim) + ".csv", sep=','
    )
    return sub_outliers['bids_name'].values.tolist()

def load_data(sub):
    """
    Load functional, onset, and confound files for a given subject.
    """
    data_path = derivatives + '/fmriprep-22.1.1/sub-' + sub + '/'
    func_file = [item for item in os.listdir(data_path + '/func/') if item.endswith('desc-preproc_bold.nii.gz')]
    func_file.sort()
        
    onset_file = [item for item in os.listdir("/neurospin/nfbd/Decoding/ds000108/rawdata/sub-" + sub + "/func/") if item.endswith('events.tsv')]
    onset_file.sort()
    
    confound_file = [item for item in func_file if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file.sort()
    
    return func_file, onset_file, confound_file, data_path

def load_event(sub, onset, func, confound, data_path, func_files, onset_list, confounds):
    """
    Load event data, functional data, and confound data for a given run.
    """
    data = pd.read_csv(main_path + dataset + '/rawdata/sub-' + sub + "/func/" + onset, sep='\t')
    onsets_tr = data[['trial_type', 'onset']].copy()
    duration = np.hstack((onsets_tr.onset.values[1:] - onsets_tr.onset.values[:-1], 5))  # Note: 5 is arbitrary
    onsets_tr['duration'] = duration
    conditions = onsets_tr[onsets_tr['trial_type'].isin(['Look_Neutral_Stim', 'Look_Neg_Stim'])]

    # Load and smooth functional data
    fmri_data = image.smooth_img(data_path + 'func/' + func, fwhm=6)
    
    # Load confounds
    confound = pd.read_csv(data_path + 'func/' + confound, sep='\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]

    func_files.append(fmri_data)
    onset_list.append(conditions)
    confounds.append(confound)
    
    return func_files, onset_list, confounds, conditions   

def first_level(func_files, onset_list, confounds):
    """
    Perform first-level GLM analysis on the functional data.
    """
    TR = 2
    glm = FirstLevelModel(
        t_r=TR,
        mask_img=mask_name,
        high_pass=1/128,
        smoothing_fwhm=None
    )
    first_level_model = glm.fit(func_files, events=onset_list, confounds=confounds)
    return first_level_model.design_matrices_, glm

def make_localizer_contrasts(design_matrix):
    """
    Create contrasts for the localizer task.
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }
    contrasts["LookNeg"] = contrasts["Look_Neg_Stim"]
    contrasts["LookNeu"] = contrasts["Look_Neutral_Stim"]
    
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
    return glm.compute_contrast(contrast_val, output_type="all")

def decodeur(contrast_imgs, conditions_label, groups, mask_name, cv):
    """
    Perform classification using FREMClassifier.
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
    Perform permutation test to assess the significance of the model's performance.
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

#%%##########################################################################
# PATH SETUP -----------------------------------------------------------
##########################################################################
main_path = '/neurospin/nfbd/Decoding/'
dataset = 'ds000108'
derivatives = main_path + dataset + '/derivatives/'

masks = ['frontal_mask.nii', 'limbic_mask.nii', 'fronto_limbic.nii', 'wb_wfu.nii']

for mask in masks:
    mask_name = main_path + 'code/mask/' + mask

    sub_file = glob.glob(derivatives + 'fmriprep-22.1.1/sub-*')
    sub_file = [item for item in sub_file if not item.endswith('.html')]

    sub_label_bmaps, conditions_label_bmaps, contrast_imgs = [], [], []

    for i in range(1, len(sub_file) + 1):
        if i in [6, 11, 25, 32]:
            continue  # Skip outlier subjects

        sub = f'{i:02d}'
        print(f'Subject: {sub}')
        
        func_files = []
        onset_list = []
        confounds = []

        func_file, onset_file, confound_file, data_path = load_data(sub)
        
        for k in range(1, len(onset_file) + 1):
            run = f'{k:02d}'
            print(f'Run: {run}')
            
            onset = onset_file[k - 1]
            func = func_file[k - 1]
            confound = confound_file[k - 1]
            func_split = '_'.join(func.split('_space-MNI152NLin2009cAsym_desc-preproc_'))
            
            sub_outliers = outliers(0.3)
            
            if func_split.split('.nii.gz')[0] in sub_outliers:
                continue
            
            print('Processing')
            func_files, onset_list, confounds, conditions = load_event(sub, onset, func, confound, data_path, func_files, onset_list, confounds)
            
        design_matrices, glm = first_level(func_files, onset_list, confounds)

        print('Step: save beta-maps')
        for i in range(len(design_matrices)):
            contrasts = make_localizer_contrasts(design_matrices[i])
            fmri_glm = glm.fit(func_files[i], design_matrices=design_matrices[i])
            
            for contrast_id, contrast_val in contrasts.items():
                if contrast_id in ['neg', 'neu']:
                    summary_statistics_run = plot_contrast(fmri_glm, contrast_val, contrast_id, i)
                    contrast_imgs.append(summary_statistics_run['z_score'])
                    conditions_label_bmaps.append(contrast_id)
                    sub_label_bmaps.append(sub)

    # Run decoder
    print('Running decoder')
    decoder_FREM = decodeur(contrast_imgs, conditions_label_bmaps, sub_label_bmaps, mask_name, StratifiedShuffleSplit(n_splits=5, random_state=42))
    weight_img = decoder_FREM.coef_img_['neg']
    # weight_img.to_filename(derivatives + 'nilearn/first_level_analysis/weights_img/' + mask.split('.')[0] + '_neg_weights.nii')

    # Permutation test
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
    coef_img = decoder_FREM_permut.coef_img_['neg']
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
    thresholded_img.to_filename(derivatives + 'nilearn/first_level_analysis/weights_img/' + mask.split('.')[0] + '_neg_weights_thr.nii')

    clusters_tb = get_clusters_table(thresholded_img, 1 - threshold)
    clusters_df = pd.DataFrame(clusters_tb)
    clusters_df.to_csv(derivatives + 'nilearn/first_level_analysis/csvfiles/contrasts_imgs_neu_vs_neg_ds000108/' + mask.split('.')[0] + '_neg_clusters_table.csv', index=False)

    # Perform permutation test for the accuracy
    actual_score, permuted_scores, pvalue = permutation_test(contrast_imgs, conditions_label_bmaps, decoder_FREM_permut, sub_label_bmaps, n_permutations=1000)

    # Save results
    print('Saving results')
    summary = {
        'cv_scores': list(decoder_FREM.cv_scores_['neg']),
        'mask': mask_name.rsplit('/', 8)[-1].rsplit('.')[0]
    }
    df2 = pd.DataFrame.from_records(summary)
    df2.to_csv(derivatives + 'nilearn/first_level_analysis/csvfiles/contrasts_imgs_neu_vs_neg_ds000108/summary_table_data_across_sub_' +
               mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + '_noerror.csv')

    # Opening a file named "permutation_score.txt" in write mode
    file = open(derivatives+'nilearn/first_level_analysis/csvfiles/contrasts_imgs_reapp_vs_neg_ds000108/permutation_score_' + 
             mask_name.rsplit('/', 8)[-1].rsplit('.')[0]+'.txt', "w")

	# lines = ['Permutation score = ' + str(np.mean(null_cv_scores)), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "\nContrast =" + list(contrasts)[1]]
    lines = ['Permutation score = ' + str(np.mean(np.array(permuted_scores))), "P-value_score = " + str(pvalue), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "\nContrast =" + list(contrasts)[0]]

    with file as f:
        for line in lines:
            f.write(line)
            f.write('\n')

