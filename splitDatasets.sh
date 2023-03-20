#!/bin/bash


python3 dataset_splitting/splitIMDBDatasets.py
python3 dataset_splitting/splitJSONIMDBDatasets.py --dataset-path datasets/original_augmented_1x_aclImdb
python3 dataset_splitting/splitYelpDatasets.py