import os

root_dir = os.path.join(os.environ['HOME'], 'OneDrive - Imperial College London')
data_root_dir = os.path.join(root_dir, 'tierpsy_features_experiments/classify/data')
results_root_dir = os.path.join(root_dir, 'tierpsy_features_experiments/classify/results')

col2ignore = ['Unnamed: 0', 'exp_name', 'id', 'base_name', 
              'date', 'worm_id', 'days_of_adulthood',
              'original_video', 'directory', 'strain',
       'strain_description', 'allele', 'gene', 'chromosome',
       'tracker', 'sex', 'developmental_stage', 'ventral_side', 'food',
       'habituation', 'experimenter', 'arena', 'exit_flag', 'experiment_id',
       'n_valid_frames', 'n_missing_frames', 'n_segmented_skeletons',
       'n_filtered_skeletons', 'n_valid_skeletons', 'n_timestamps',
       'first_skel_frame', 'last_skel_frame', 'fps', 'total_time',
       'microns_per_pixel', 'mask_file_sizeMB', 'skel_file', 'frac_valid',
       'worm_index', 'n_frames', 'n_valid_skel', 'first_frame']

