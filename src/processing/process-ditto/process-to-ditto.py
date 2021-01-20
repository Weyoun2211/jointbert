import pandas as pd
import numpy as np
np.random.seed(42)
import random
random.seed(42)

from tqdm.notebook import tqdm

from pathlib import Path
import jsonlines

Path('../../../data/processed/wdc-lspc/ditto/jsonlines/').mkdir(parents=True, exist_ok=True)

def extract_ditto_sequence(row):
    row['ditto_left'] = f'COL brand VAL {row["brand_left"]} COL title VAL {row["title_left"]} COL description VAL {row["description_left"]} COL specTable VAL {row["specTableContent_left"]}'
    row['ditto_right'] = f'COL brand VAL {row["brand_right"]} COL title VAL {row["title_right"]} COL description VAL {row["description_right"]} COL specTable VAL {row["specTableContent_right"]}'
    row['ditto_left_titleonly'] = f'COL brand VAL {row["brand_left"]} COL title VAL {row["title_left"]}'
    row['ditto_right_titleonly'] = f'COL brand VAL {row["brand_right"]} COL title VAL {row["title_right"]}'
    return row

def process_to_ditto(dataset):
    tqdm.pandas(desc="Extracting Ditto Sequences")
    dataset = dataset.progress_apply(extract_ditto_sequence, axis=1)
    return dataset


datasets_lspc_train = ['preprocessed_computers_train_small', 'preprocessed_computers_train_medium',
                       'preprocessed_computers_train_large', 'preprocessed_computers_train_xlarge',
                       'preprocessed_cameras_train_small', 'preprocessed_cameras_train_medium',
                       'preprocessed_cameras_train_large', 'preprocessed_cameras_train_xlarge',
                       'preprocessed_watches_train_small', 'preprocessed_watches_train_medium',
                       'preprocessed_watches_train_large', 'preprocessed_watches_train_xlarge',
                       'preprocessed_shoes_train_small', 'preprocessed_shoes_train_medium',
                       'preprocessed_shoes_train_large', 'preprocessed_shoes_train_xlarge'
                       ]

datasets_lspc_gs = ['preprocessed_computers_gs', 'preprocessed_cameras_gs',
                    'preprocessed_watches_gs', 'preprocessed_shoes_gs', 'preprocessed_computers_new_testset_1500'
                    ]

for ds in datasets_lspc_gs:
    dataset = pd.read_pickle(f'../../../data/interim/wdc-lspc/gold-standards/{ds}.pkl.gz')
    dataset = dataset.fillna('')

    processed = process_to_ditto(dataset)

    with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}.jsonl', mode='w') as writer:
        for i, row in processed.iterrows():
            row_left = row[['brand_left', 'title_left', 'description_left', 'specTableContent_left']]
            columns = row_left.index
            columns = [x.replace('_left', '') for x in columns]
            row_left.index = columns

            row_right = row[['brand_right', 'title_right', 'description_right', 'specTableContent_right']]
            columns = row_right.index
            columns = [x.replace('_right', '') for x in columns]
            row_right.index = columns

            writer.write([row_left.to_dict(), row_right.to_dict()])

    with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}_titleonly.jsonl', mode='w') as writer:
        for i, row in processed.iterrows():
            row_left = row[['brand_left', 'title_left']]
            columns = row_left.index
            columns = [x.replace('_left', '') for x in columns]
            row_left.index = columns

            row_right = row[['brand_right', 'title_right']]
            columns = row_right.index
            columns = [x.replace('_right', '') for x in columns]
            row_right.index = columns

            writer.write([row_left.to_dict(), row_right.to_dict()])

    with open(f'../../../data/processed/wdc-lspc/ditto/{ds}.txt', 'w') as file_object:
        for i, row in processed.iterrows():
            file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

    with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_titleonly.txt', 'w') as file_object:
        for i, row in processed.iterrows():
            file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

for ds in datasets_lspc_train:
    dataset = pd.read_pickle(f'../../../data/interim/wdc-lspc/training-sets/{ds}.pkl.gz')
    dataset = dataset.fillna('')
    dataset = dataset.set_index('pair_id', drop=False)

    processed = process_to_ditto(dataset)

    filename_split = ds.split('_')
    valid_name = f'{filename_split[1]}_valid_{filename_split[3]}'
    valid_ids = pd.read_csv(f'../../../data/raw/wdc-lspc/validation-sets/{valid_name}.csv')
    id_list = valid_ids['pair_id'].to_list()

    valid_set = processed.loc[id_list].copy()
    train_set = processed.drop(id_list)

    with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}.jsonl', mode='w') as writer:
        for i, row in train_set.iterrows():
            row_left = row[['brand_left', 'title_left', 'description_left', 'specTableContent_left']]
            columns = row_left.index
            columns = [x.replace('_left', '') for x in columns]
            row_left.index = columns

            row_right = row[['brand_right', 'title_right', 'description_right', 'specTableContent_right']]
            columns = row_right.index
            columns = [x.replace('_right', '') for x in columns]
            row_right.index = columns

            writer.write([row_left.to_dict(), row_right.to_dict()])

    with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{ds}_titleonly.jsonl', mode='w') as writer:
        for i, row in train_set.iterrows():
            row_left = row[['brand_left', 'title_left']]
            columns = row_left.index
            columns = [x.replace('_left', '') for x in columns]
            row_left.index = columns

            row_right = row[['brand_right', 'title_right']]
            columns = row_right.index
            columns = [x.replace('_right', '') for x in columns]
            row_right.index = columns

            writer.write([row_left.to_dict(), row_right.to_dict()])

    with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{valid_name}.jsonl', mode='w') as writer:
        for i, row in valid_set.iterrows():
            row_left = row[['brand_left', 'title_left', 'description_left', 'specTableContent_left']]
            columns = row_left.index
            columns = [x.replace('_left', '') for x in columns]
            row_left.index = columns

            row_right = row[['brand_right', 'title_right', 'description_right', 'specTableContent_right']]
            columns = row_right.index
            columns = [x.replace('_right', '') for x in columns]
            row_right.index = columns

            writer.write([row_left.to_dict(), row_right.to_dict()])

    with jsonlines.open(f'../../../data/processed/wdc-lspc/ditto/jsonlines/{valid_name}_titleonly.jsonl',
                        mode='w') as writer:
        for i, row in valid_set.iterrows():
            row_left = row[['brand_left', 'title_left']]
            columns = row_left.index
            columns = [x.replace('_left', '') for x in columns]
            row_left.index = columns

            row_right = row[['brand_right', 'title_right']]
            columns = row_right.index
            columns = [x.replace('_right', '') for x in columns]
            row_right.index = columns

            writer.write([row_left.to_dict(), row_right.to_dict()])

    with open(f'../../../data/processed/wdc-lspc/ditto/{ds}.txt', 'w') as file_object:
        for i, row in train_set.iterrows():
            file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

    with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_titleonly.txt', 'w') as file_object:
        for i, row in train_set.iterrows():
            file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

    with open(f'../../../data/processed/wdc-lspc/ditto/{valid_name}.txt', 'w') as file_object:
        for i, row in valid_set.iterrows():
            file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

    with open(f'../../../data/processed/wdc-lspc/ditto/{valid_name}_titleonly.txt', 'w') as file_object:
        for i, row in valid_set.iterrows():
            file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')

    with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_full.txt', 'w') as file_object:
        for i, row in processed.iterrows():
            file_object.write(f'{row["ditto_left"]}\t{row["ditto_right"]}\t{row["label"]}\n')

    with open(f'../../../data/processed/wdc-lspc/ditto/{ds}_titleonly_full.txt', 'w') as file_object:
        for i, row in processed.iterrows():
            file_object.write(f'{row["ditto_left_titleonly"]}\t{row["ditto_right_titleonly"]}\t{row["label"]}\n')