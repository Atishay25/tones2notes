#!/bin/bash

DATASET_DIR="../data/MAPS"
WORKSPACE=$(pwd)

python3 features.py --dir $DATASET_DIR --workspace $WORKSPACE