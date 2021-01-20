import numpy as np
np.random.seed(42)
import random
random.seed(42)

import glob


from src.models.deepmatcher import run_deepmatcher


from sys import argv

gpu_id = argv[1]

train_sizes = {'xlarge':9, 'large':6, 'medium':4, 'small':3}
lr_range = [0.0005, 0.0008, 0.001, 0.003, 0.005]
feature_combinations = [['title', 'description', 'brand', 'specTableContent'], ['brand', 'title']]

for feature_combination in feature_combinations:
    for learning_rate in lr_range:
        for train_size, value in train_sizes.items():
            for file in glob.glob('../../../data/processed/wdc-lspc/magellan/learning-curve/formatted/*'):
                if f'shoes_trainonly_{train_size}' in file and 'metadata' not in file and '_pairs_' in file:

                    train_set = file
                    valid_set = file.replace('trainonly','valid')
                    test_set = '../../../data/processed/wdc-lspc/magellan/learning-curve/formatted/preprocessed_shoes_gs_magellan_pairs_formatted.csv'
                    pred_set = [
                        '../../../data/processed/wdc-lspc/magellan/learning-curve/formatted/preprocessed_shoes_gs_magellan_pairs_formatted.csv']

                    experiment_name = 'wdc-lspc'
                    epochs = 50
                    pos_neg_ratio = value
                    batch_size = 16
                    lr = learning_rate
                    lr_decay = 0.8
                    embedding = 'fasttext.en.bin'
                    nn_type = 'rnn'
                    comp_type = 'abs-diff'
                    special_name = 'standard'
                    features = feature_combination

                    run_deepmatcher.run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio,
                                                 batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features,
                                                 1, prediction_sets=pred_set)
                    run_deepmatcher.run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio,
                                                 batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features,
                                                 2, prediction_sets=pred_set)
                    run_deepmatcher.run_dm_model(train_set, valid_set, test_set, experiment_name, gpu_id, epochs, pos_neg_ratio,
                                                 batch_size, lr, lr_decay, embedding, nn_type, comp_type, special_name, features,
                                                 3, prediction_sets=pred_set)
