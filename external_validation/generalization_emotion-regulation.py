#%%##### IMPORT --------------------------------------------------------------
import os, glob
import pandas as pd
import numpy as np
from nilearn import image
from nilearn.decoding import FREMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score
from nilearn.glm.first_level import FirstLevelModel
from sklearn.utils import shuffle
from nilearn.image import new_img_like
from nilearn.reporting import get_clusters_table

#%%##### PATH ---------------------------------------------------------------
main_path = '/neurospin/nfbd/Decoding/'

# ds000108
dataset_ds000108 = 'ds000108'
derivatives_ds000108 = main_path + dataset_ds000108 + '/derivatives/'
mriqc_path_ds000108 = derivatives_ds000108 +'mriqc-22.0.6'
sub_file_ds000108 = glob.glob(derivatives_ds000108 + 'fmriprep-22.1.1/sub-*')
sub_file_ds000108 = [item for item in sub_file_ds000108 if not item.endswith('.html')]

# ds003548
dataset_ds003548 = 'ds003548'
derivatives_ds003548 = main_path + dataset_ds003548 + '/derivatives/'
sub_file_ds003548 = glob.glob(derivatives_ds003548 + 'fmriprep-22.1.1/sub-*')
sub_file_ds003548 = [item for item in sub_file_ds003548 if not item.endswith('.html')]

# ds003823
dataset_ds003823 = 'ds003823'
derivatives_ds003823 = main_path + dataset_ds003823 + '/derivatives/'
mriqc_path_ds003823 = derivatives_ds003823 + 'mriqc'
sub_file_ds003823 = [item for item in os.listdir(derivatives_ds003823 + 'fmriprep-22.1.1/') if not item.endswith('.html') and item.startswith('sub-')]

# mask
masks = ['frontal_mask.nii', 'limbic_mask.nii', 'fronto_limbic.nii', 'wb_wfu.nii']

#%%#####  FUNCTION

# function to find outliers from quality check
def outliers(fd_lim, mriqc_path, column):
    sub_outliers = pd.read_csv(mriqc_path + '/name_suboutliers_mriqc_fd-lim-' + str(fd_lim) + ".csv", sep = ',')
    sub_outliers = sub_outliers[column].values.tolist()
    
    return sub_outliers

# load data for Wager et al., 2008
def load_data_ds000108(sub, derivatives, dataset):
    data_path_ds000108 = derivatives + '/fmriprep-22.1.1/sub-' + sub + '/'
    path_func = os.listdir(data_path_ds000108 + '/func/' )
    func_file_ds000108 = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
    func_file_ds000108.sort()
        
    path_onsets = os.listdir("/neurospin/nfbd/Decoding/" + dataset + "/rawdata/sub-" + sub + "/func/")
    onset_file_ds000108 = [item for item in path_onsets if item.endswith('events.tsv')]
    onset_file_ds000108.sort()
    
    confound_file_ds000108 = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file_ds000108.sort()
    
    return func_file_ds000108, onset_file_ds000108, confound_file_ds000108, data_path_ds000108

# load data for Nashiro et al., 2021
def load_data_ds003548(sub, derivatives, dataset):
    data_path_ds003548 = derivatives + 'fmriprep-22.1.1/sub-' + sub + '/'
    path_func = os.listdir(data_path_ds003548 + '/func/' )
    func_file_ds003548 = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
    func_file_ds003548.sort()
        
    path_onsets = os.listdir ("/neurospin/nfbd/Decoding/" + dataset + "/rawdata/")
    onset_file_ds003548 = [item for item in path_onsets if item.endswith('events.tsv')]
    onset_file_ds003548.sort()
    
    confound_file_ds003548 = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file_ds003548.sort()
    
    return func_file_ds003548, onset_file_ds003548, confound_file_ds003548, data_path_ds003548

# load data for David et al., 2021
def load_data_ds003823(sub, derivatives, dataset):
    data_path_ds003823 = derivatives + 'fmriprep-22.1.1/' + sub 
    path_func = os.listdir(data_path_ds003823 + '/ses-pre/func/' )
    func_file_ds003823 = [item for item in path_func if item.endswith('desc-preproc_bold.nii.gz')]
    func_file_ds003823.sort()
        
    path_onsets = os.listdir("/neurospin/nfbd/Decoding/" + dataset + "/rawdata/" + sub + "/ses-pre/func/")
    onset_file_ds003823 = [item for item in path_onsets if item.endswith('emotionRegulation_events.tsv')]
    onset_file_ds003823.sort()
    
    confound_file_ds003823 = [item for item in path_func if item.endswith('desc-confounds_timeseries.tsv')]
    confound_file_ds003823.sort()
    
    return func_file_ds003823, onset_file_ds003823, confound_file_ds003823, data_path_ds003823

#load csv event Wager et al., 2008
def load_event_ds000108(dataset, sub, onset, func_ds000108, confound_ds000108, data_path, 
                        func_files_ds000108, onset_list_ds000108, confounds_ds000108):
    
    data = pd.read_csv(main_path + dataset + '/rawdata/sub-' + sub + "/func/" + onset, sep = '\t')
    
    onsets_tr_a = data[['trial_type', 'onset']]
    onsets_tr = onsets_tr_a.copy()
    duration = np.hstack((onsets_tr.onset.values[1:] - onsets_tr.onset.values[:-1], 5))
    
    # Note the 5 at the end is arbitrary (fixme)
    onsets_tr['duration'] = duration
    
    conditions_ds000108 = onsets_tr[onsets_tr['trial_type'].isin([['Reapp_Neg_Stim', 'Look_Neg_Stim'])]

    func_path = data_path + 'func/'
    fmri_data = image.smooth_img(func_path + func_ds000108, fwhm=6) #☺for the run 1
    
    confound = pd.read_csv(func_path + confound_ds000108, sep = '\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]

    func_files_ds000108.append(fmri_data)
    onset_list_ds000108.append(conditions_ds000108)
    confounds_ds000108.append(confound)
    
    # conditions = conditions.trial_type.unique() 
    
    return func_files_ds000108, onset_list_ds000108, confounds_ds000108, conditions_ds000108   
 

# load csv event David et al., 2021
def load_event_ds003823(dataset, sub, func_ds003823, confound_ds003823, data_path, func_files_ds003823, 
                        onset_list_ds003823, confounds_ds003823):
    
    data = pd.read_csv("/neurospin/nfbd/Decoding/"+ dataset +"/rawdata/" + sub + "/ses-pre/func/" + 
                       sub + "_ses-pre_task-emotionRegulation_events.tsv", sep = '\t')
    
    # Keep only columns trial_type, onset and duration
    onsets_tr_a = data[['trial_type', 'onset','duration']]   
    onsets_tr = onsets_tr_a.copy()
     
    conditions_ds003823 = onsets_tr[onsets_tr['trial_type'].isin(['dn', 'vn'])]#create a new dataframe with only the condition of interest
   
    # LOAD FUNCTIONAL DATA --------------------------------------------

    func = data_path + '/ses-pre/func/'
    fmri_data = image.smooth_img(func + func_ds003823[0], fwhm=6) #for the run 1
    
    #load confounds
    confound = pd.read_csv(func + confound_ds003823[0], sep = '\t')
    reg_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound = confound[reg_names]

    func_files_ds003823.append(fmri_data)
    onset_list_ds003823.append(conditions_ds003823)
    confounds_ds003823.append(confound)
    
    # conditions_ds003823 = conditions.trial_type.unique() 
    
    return func_files_ds003823, onset_list_ds003823, confounds_ds003823, conditions_ds003823

# first level analysis Wager et al., 2008
def first_level_ds000108(func_files_ds000108, onset_list_ds000108, confounds_ds000108):
    TR = 2
    
    glm_ds000108 = FirstLevelModel(
         t_r=TR,
         mask_img = mask_name,
         high_pass=0.008,
         smoothing_fwhm=None)
        # memory="nilearn_cache")
        
    first_level_model = glm_ds000108.fit(func_files_ds000108,events=onset_list_ds000108, 
                                confounds = confounds_ds000108) #add confounds
    
    # design_matrix = first_level_model.design_matrices_[0]
    design_matrices_ds000108 = first_level_model.design_matrices_
    
    return design_matrices_ds000108, glm_ds000108


# first level analysis David et al., 2021
def first_level_ds003823(func_files_ds003823, onset_list_ds003823, confounds_ds003823):
    TR = 2.4
    
    glm_ds003823 = FirstLevelModel(
         t_r=TR,
         mask_img = mask_name,
         high_pass=0.008,
         smoothing_fwhm=None)
        # memory="nilearn_cache")
        
    first_level_model = glm_ds003823.fit(func_files_ds003823,events=onset_list_ds003823, 
                                         confounds = confounds_ds003823) #add confounds
    
    # design_matrix = first_level_model.design_matrices_[0]
    design_matrices_ds003823 = first_level_model.design_matrices_
    
    return design_matrices_ds003823, glm_ds003823

# contrasts Wager et al., 2008
def make_localizer_contrasts_ds000108(design_matrix):
       contrast_matrix = np.eye(design_matrix.shape[1])
       contrasts_ds000108 = {
              column: contrast_matrix[i]
              for i, column in enumerate(design_matrix.columns)
          }
          
       
       # Short dictionary of more relevant contrasts
       contrasts_ds000108 = {
           # "neg - neu": contrasts["LookNeg"] - contrasts["LookNeu"],
           # "neu - neg": contrasts["LookNeu"] - contrasts["LookNeg"],
           "neu" : contrasts_ds000108["Look_Neutral_Stim"],
           "neg" : contrasts_ds000108["Look_Neg_Stim"]}
       
       return contrasts_ds000108
   

# contrasts David et al., 2021
def make_localizer_contrasts_ds003823(design_matrix):
       contrast_matrix = np.eye(design_matrix.shape[1])
       contrasts_ds003823 = {
              column: contrast_matrix[i]
              for i, column in enumerate(design_matrix.columns)
          }          
       
       contrasts_ds003823["LookNeg"] = contrasts_ds003823["vn"]
       contrasts_ds003823["LookNeu"] = contrasts_ds003823["vt"]
       
       
       # Short dictionary of more relevant contrasts
       contrasts_ds003823 = {
           # "neg - neu": contrasts_ds003823["LookNeg"] - contrasts_ds003823["LookNeu"],
           # "neu - neg": contrasts_ds003823["LookNeu"] - contrasts_ds003823["LookNeg"],
           "neu" : contrasts_ds003823["LookNeu"],
           "neg" : contrasts_ds003823["LookNeg"]}
       
       return contrasts_ds003823
                        
                        
def plot_contrast_ds000108(glm, contrast_val, contrast_id, nb_run):
    """Specify, estimate and plot the main contrasts \
        for given a first model."""

    # compute the per-contrast z-map
    summary_statistics_run_ds000108 = glm.compute_contrast(
        contrast_val, output_type="all"
    )
   
    return summary_statistics_run_ds000108

def plot_contrast_ds003823(glm, contrast_val, contrast_id, nb_run):
    """Specify, estimate and plot the main contrasts \
        for given a first model."""

    # compute the per-contrast z-map
    summary_statistics_run_ds003823 = glm.compute_contrast(
        contrast_val, output_type="all"
    )
   
    return summary_statistics_run_ds003823
    
def decodeur(mask_name, cv):
    
    decoder_FREM = FREMClassifier(estimator='svc', mask=mask_name, cv=cv, #GroupKFold(n_splits=6)
                              clustering_percentile = 10, screening_percentile=20,
                              scoring='roc_auc', 
                              smoothing_fwhm=4,standardize=False,verbose =3) #smoothing ??
    
    
    # decoder_FREM.fit(contrast_imgs, conditions_label,groups = groups)
    
    return decoder_FREM

def permutation_test(X, y, model, groups, n_permutations=1000):
    """
    Perform permutation test to evaluate the significance of the model's performance.
    """
    model.fit(X, y, groups=groups)
    y_pred = model.predict(X)
    actual_score = np.mean(list(model.cv_scores_['reg']))
    permuted_scores = []
    
    for _ in range(n_permutations):
        y_permuted = shuffle(y)
        model.fit(X, y_permuted, groups=groups)
        y_permuted_pred = model.predict(X)
        permuted_score = accuracy_score(y, y_permuted_pred)
        permuted_scores.append(permuted_score)
    
    p_value = (np.sum(np.array(permuted_scores) >= actual_score) + 1) / (n_permutations + 1)
    return actual_score, permuted_scores, p_value

#%%####### Empty lists -------------------------------------------------------
sub_label_bmaps_ds003823 = []
sub_label_bmaps_ds000108 = []
conditions_label_bmaps_ds003823 = []
conditions_label_bmaps_ds000108 = []
contrast_imgs_ds003823 = []
contrast_imgs_ds000108 = []

cut_coords = [-27,-4,-20] # amygdala coordinates
#%%####### TRAIN -------------------------------------------------------------
for mask in masks:
    mask_name = main_path + 'code/mask/' + mask
    
    print('Train... ds003823')
    for idx, i in enumerate(sub_file_ds003823):
        sub = sub_file_ds003823[idx]
        sub_outliers = outliers(0.3,mriqc_path_ds003823, 'participant_id')
        
        if sub in sub_outliers or sub == 'sub-7007':
            pass
        else: 
            print('Subject :' + sub)
            # os.makedirs(derivatives+'nilearn/output/' + sub, exist_ok=True)
            
            
            # LOAD DATA FILES ------------------------------------------------
            print('Load data files..')
            
            # empty lists
            func_files_ds003823, onset_list_ds003823, confounds_ds003823 = [],[],[]
        
            # load data 
            func_file_ds003823, onset_file_ds003823, confound_file_ds003823, data_path_ds003823 = load_data_ds003823(sub, derivatives_ds003823, dataset_ds003823)
            
            # load event
            func_files_ds003823, onset_list_ds003823, confounds_ds003823, conditions_ds003823 = load_event_ds003823(dataset_ds003823, sub, 
                                                                                                                    func_file_ds003823, confound_file_ds003823, data_path_ds003823, 
                                                                                                                    func_files_ds003823, onset_list_ds003823, 
                                                                                                                    confounds_ds003823)
            
            # first level analysis
            design_matrices_ds003823, glm_ds003823 = first_level_ds003823(func_files_ds003823, onset_list_ds003823, confounds_ds003823)
        
            print('Step : save beta-maps')
            for i in range(0,len(design_matrices_ds003823)):
            
                 # Call the contrast specification within the function
                contrasts_ds003823 = make_localizer_contrasts_ds003823(design_matrices_ds003823[i])
                
                # fit the glm
                fmri_glm = glm_ds003823.fit(func_files_ds003823[i], design_matrices=design_matrices_ds003823[i])
                
                for index, (contrast_id, contrast_val) in enumerate(contrasts_ds003823.items()):
                    
                    if contrast_id == 'neg' or contrast_id == 'reg':

                        # plot constrasts
                        summary_statistics_run_ds003823 = plot_contrast_ds003823(fmri_glm, contrast_val,
                                                               contrast_id, i)
                        
                        contrast_imgs_ds003823.append(summary_statistics_run_ds003823['z_score'])
                        conditions_label_bmaps_ds003823.append(contrast_id)
                        sub_label_bmaps_ds003823.append(sub)


    #%%####### TEST DS000108 -----------------------------------------------------

    print('Test... ds000108')
    for i in range(1,len(sub_file_ds000108)+1):
        if i == 6 or i == 11 or i == 12 or i == 25 or i == 32:
            continue 
        
        sub = '%02d' % i
        print('Subject:' + sub)
        
        # empty list
        func_files_ds000108, onset_list_ds000108, confounds_ds000108 = [], [], []
        
        # load data
        func_file_ds000108, onset_file_ds000108, confound_file_ds000108, data_path_ds000108 = load_data_ds000108(sub, derivatives_ds000108, dataset_ds000108)

        # loop over runs
        for k in range(1,len(onset_file_ds000108)+1): #+1 car le script ne faisait pas tous les runs
            run = '%02d' % k
            print('Run:' + run)
            
            onset_ds000108 = onset_file_ds000108[int(run)-1] # get the specific onset_file for the run
            func_ds000108 = func_file_ds000108[int(run)-1] # get the specific func_file for the run
            confound_ds000108 = confound_file_ds000108[int(run)-1]
            func_split = '_'.join(func_ds000108.split('_space-MNI152NLin2009cAsym_desc-preproc_')) # get only the name of the file
            
            sub_outliers = outliers(0.3,mriqc_path_ds000108, 'bids_name') # load sub_name 
            
            if func_split.split('.nii.gz')[0] in sub_outliers:
                pass
            else: 
                print('ok')
                
                func_files_ds000108, onset_list_ds000108, confounds_ds000108, conditions_ds000108 = load_event_ds000108(dataset_ds000108, sub, onset_ds000108, 
                                                                                                                        func_ds000108, confound_ds000108, data_path_ds000108, func_files_ds000108, 
                                                                                                                        onset_list_ds000108, confounds_ds000108)
                
        design_matrices_ds000108, glm_ds000108 = first_level_ds000108(func_files_ds000108, onset_list_ds000108, confounds_ds000108)
    

        print('Step : save beta-maps')
        for i in range(0,len(design_matrices_ds000108)):
        
             # Call the contrast specification within the function    
            contrasts_ds000108 = make_localizer_contrasts_ds000108(design_matrices_ds000108[i])
            fmri_glm = glm_ds000108.fit(func_files_ds000108[i], design_matrices=design_matrices_ds000108[i])
            
            for index, (contrast_id, contrast_val) in enumerate(contrasts_ds000108.items()):
                
                if contrast_id == 'neg' or contrast_id == 'reg':

                    summary_statistics_run_ds000108 = plot_contrast_ds000108(fmri_glm, contrast_val,
                                                           contrast_id, i)
                    
                    contrast_imgs_ds000108.append(summary_statistics_run_ds000108['z_score'])
                    conditions_label_bmaps_ds000108.append(contrast_id)
                    sub_label_bmaps_ds000108.append(sub)
            
             
                    
    #%%###### DECODEUR 
    decoder_FREM = decodeur(mask_name, StratifiedShuffleSplit(n_splits = 5, random_state =42)) #quel cv ?

    print('decoder fit')
    decoder_FREM.fit(contrast_imgs_ds003823,conditions_label_bmaps_ds003823)

    print('decoder predict ds000108')
    ds000108_prob = decoder_FREM.predict(contrast_imgs_ds000108)

    ############################################################################
    # PERMUTATION TEST ----------------------------------
    ######################################################################        
    # Test performed with ds000108
    decoder_FREM_permut_ds003823 = FREMClassifier(
        estimator='svc',
        mask=mask_name,
        cv=StratifiedShuffleSplit(n_splits = 5, random_state =42),
        clustering_percentile=10,
        screening_percentile=20,
        scoring='accuracy',
        smoothing_fwhm=4,
        standardize=True,
        verbose=3
    )
    
    # permutation test for Wager et al., 2008
    decoder_FREM_permut_ds003823.fit(contrast_imgs_ds003823, conditions_label_bmaps_ds000108, groups=sub_label_bmaps_ds003823)
    
    # Permutation test for accuracy
    actual_score_ds000108, permuted_scores_ds000108, pvalue_ds000108 = permutation_test(contrast_imgs_ds000108, conditions_label_bmaps_ds000108, decoder_FREM_permut_ds003823, sub_label_bmaps_ds000108, n_permutations=1000)
    
    # Opening a file named "permutation_score.txt" in write mode
    print('save permutation score')
    file = open('/neurospin/nfbd/Decoding/output/test-ds000108_permutation_score_' + 
           mask_name.rsplit('/', 8)[-1].rsplit('.')[0]+'_regneg.txt', "w")


    # lines = ['Permutation score = ' + str(np.mean(null_cv_scores)), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "\nContrast =" + list(contrasts)[1]]
    lines = ['Permutation score = ' + str(np.mean(np.array(permuted_scores_ds000108))), "P-value_score = " + str(pvalue_ds000108), "Mask_name = "+ mask_name.rsplit('/', 8)[-1].rsplit('.')[0] + "_regneg"]

    with file as f:
        for line in lines:
            f.write(line)
            f.write('\n')
            
    #%% Classification report

    # classification report ds000108
    cls_report_ds000108 = classification_report(conditions_label_bmaps_ds000108, ds000108_prob, target_names=['neg','reg'],output_dict=True)
    # print(cls_report)

    df = pd.DataFrame(cls_report_ds000108).transpose()
    df.to_csv('/neurospin/nfbd/Decoding/output/test-ds000108_classification_report_' + mask_name.rsplit('/', 8)[-1].rsplit('.')[0]+'.csv')

   
