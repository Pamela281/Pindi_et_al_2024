#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:37:24 2024

@author: pp262170
"""

#%%##### IMPORTS
import os
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

#%%##### FUNCTION DEFINITIONS

def make_localizer_contrasts(design_matrix):
    """
    Define contrasts for localizer task based on the design matrix.
    """
    contrast_matrix = np.eye(design_matrix.shape[1])
    contrasts = {column: contrast_matrix[i] for i, column in enumerate(design_matrix.columns)}
    
    contrasts["LookNeg"] = contrasts["vn"]
    contrasts["LookNeu"] = contrasts["vt"]
    
    # Define more relevant contrasts
    contrasts = {
        "neg - neu": contrasts["LookNeg"] - contrasts["LookNeu"],
        "neu - neg": contrasts["LookNeu"] - contrasts["LookNeg"],
        "neu": contrasts["LookNeu"],
        "neg": contrasts["LookNeg"]
    }
    
    return contrasts

def plot_contrast(glm, contrast_val, contrast_id):
    """
    Compute and return the contrast map for a given GLM model.
    """
    summary_statistics_run = glm.compute_contrast(contrast_val, output_type="all")
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

#%%##### MAIN SCRIPT

# Paths and mask files
main_path = '/neurospin/nfbd/Decoding/'
dataset = 'ds003823'
derivatives = os.path.join(main_path, dataset, 'derivatives/')
sub_file = [item for item in os.listdir(os.path.join(derivatives, 'fmriprep-22.1.1/')) if not item.endswith('.html') and item.startswith('sub-')]

masks = ['frontal_mask.nii', 'limbic_mask.nii', 'fronto_limbic.nii', 'wb_wfu.nii',
         'wb-wfu_without-frontal','wb-wfu_without-fronto-limbic','wb-wfu_without-limbic']]

# Load outliers
sub_outliers = pd.read_csv(os.path.join(main_path, dataset, 'derivatives', 'mriqc', 'name_suboutliers_mriqc_fd-lim-0.3.csv'), sep=',')
sub_outliers = sub_outliers['participant_id'].values.tolist()

# Loop over masks
for mask in masks:
    mask_name = os.path.join(main_path, 'code', 'mask', mask)

    sub_label_bmaps, conditions_label_bmaps, contrast_imgs = [], [], []

    for sub in sub_file:
        if sub in sub_outliers or sub == 'sub-7007':
            continue
        
        print(f'Subject: {sub}')
        
        # Load file paths
        data_path = os.path.join(derivatives, 'fmriprep-22.1.1', sub)
        path_func = os.listdir(os.path.join(data_path, 'ses-pre', 'func'))
        func_file = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
        func_file.sort()
        
        path_onsets = os.listdir(os.path.join(main_path, dataset, 'rawdata', sub, 'ses-pre', 'func'))
        onset_file = [item for item in path_onsets if item.endswith('emotionRegulation_events.tsv')]
        onset_file.sort()
        
        confound_file = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
        confound_file.sort()
        
        # Load events and confounds
        data = pd.read_csv(os.path.join(main_path, dataset, 'rawdata', sub, 'ses-pre', 'func', f'{sub}_ses-pre_task-emotionRegulation_events.tsv'), sep='\t')
        onsets_tr = data[['trial_type', 'onset', 'duration']]
        conditions = onsets_tr[onsets_tr['trial_type'].isin(['vn', 'vt'])]
        
        # Load and smooth functional data
        func = os.path.join(data_path, 'ses-pre', 'func')
        fmri_data = image.smooth_img(os.path.join(func, func_file[0]), fwhm=6)
        
        # Load confounds
        confound = pd.read_csv(os.path.join(func, confound_file[0]), sep='\t')
        reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
        confound = confound[reg_names]
        
        func_files = [fmri_data]
        onset_list = [conditions]
        confounds = [confound]
        
        # GLM First Level
        TR = 2.4
        glm = FirstLevelModel(
            t_r=TR,
            mask_img=mask_name,
            high_pass=1/128,
            smoothing_fwhm=None
        )
        first_level_model = glm.fit(func_files, events=onset_list, confounds=confounds)
        design_matrix = first_level_model.design_matrices_[0]
        
        # Define contrasts
        contrasts = make_localizer_contrasts(design_matrix)
        
        # Fit GLM and compute contrasts
        fmri_glm = glm.fit(run_imgs=func_files[0], design_matrices=design_matrix)
        
        for contrast_id, contrast_val in contrasts.items():
            if contrast_id in ['neg', 'neu']:
                summary_statistics_run = plot_contrast(fmri_glm, contrast_val, contrast_id)
                contrast_imgs.append(summary_statistics_run['z_score'])
                conditions_label_bmaps.append(contrast_id)
                sub_label_bmaps.append(sub)

    # Run decoder
    cv = StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=0.2)
    decoder_FREM = decodeur(contrast_imgs, conditions_label_bmaps, sub_label_bmaps, mask_name, cv)
    
    # Save weight image
    weight_img = decoder_FREM.coef_img_['neg']
    weight_img.to_filename(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'weights_img', f'{mask.split(".")[0]}_neg_weights.nii'))

    # Permutation test
    decoder_FREM_permut = FREMClassifier(
        estimator='svc',
        mask=mask_name,
        cv=cv,
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
        print(f'Permutations: {i+1}/{n_permutations}')
        y_permuted = shuffle(conditions_label_bmaps)
        decoder_FREM_permut.fit(contrast_imgs, y_permuted, groups=sub_label_bmaps)
        permuted_score = decoder_FREM_permut.score(contrast_imgs, y_permuted)
        null_distributions.append(permuted_score)
    
    threshold = np.percentile(null_distributions, 95)
    thresholded_data = np.where(np.abs(coef_img.get_fdata()) > 1 - threshold, coef_img.get_fdata(), 0)
    
    # Create and save thresholded image
    thresholded_img = new_img_like(coef_img, thresholded_data)
    thresholded_img.to_filename(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'weights_img', f'{mask.split(".")[0]}_neg_weights_thr.nii'))
    
    # Save clusters table
    clusters_tb = get_clusters_table(thresholded_img, 1 - threshold)
    clusters_df = pd.DataFrame(clusters_tb)
    clusters_df.to_csv(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'csvfiles', 'contrasts_imgs_neg_vs_neu_ds003823', f'{mask.split(".")[0]}_neg_clusters_table.csv'), index=False)
    
    # Permutation test for accuracy
    actual_score, permuted_scores, pvalue = permutation_test(contrast_imgs, conditions_label_bmaps, decoder_FREM_permut, sub_label_bmaps, n_permutations=1000)
    
    # Save summary and permutation scores
    summary = {
        'cv_scores': list(decoder_FREM.cv_scores_['neg']),
        'mask': mask.split('.')[0]
    }
    
    df2 = pd.DataFrame.from_records(summary)
    df2.to_csv(os.path.join(derivatives, 'nilearn', 'first_level_analysis', 'csvfiles', 'contrasts_imgs_neg_vs_neu_ds003823', f'summary_table_data_across_sub_{mask.split(".")[0]}_noerror.csv'))
    
    # Opening a file named "permutation_score.txt" in write mode
    file = open(derivatives+'nilearn/first_level_analysis/csvfiles/contrasts_imgs_neg_vs_neu_ds003823/permutation_score_' + 
               mask_name.rsplit('/', 8)[-1].rsplit('.')[0]+'.txt', "w")

    # lines = ['Permutation score = ' + str(np.mean(null_cv_scores)), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "\nContrast =" + list(contrasts)[1]]
    lines = ['Permutation score = ' + str(np.mean(np.array(permuted_scores))), "P-value_score = " + str(pvalue), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "\nContrast =" + list(contrasts)[0]]

    with file as f:
        for line in lines:
            f.write(line)
            f.write('\n')
