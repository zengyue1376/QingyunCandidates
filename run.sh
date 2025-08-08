#!/bin/bash
# for pretraining of S3Rec

# show ${RUNTIME_SCRIPT_DIR}
echo ${RUNTIME_SCRIPT_DIR}
# enter train workspace
cd ${RUNTIME_SCRIPT_DIR}/models/S3Rec

# write your code below
python -u run_pretrain.py
