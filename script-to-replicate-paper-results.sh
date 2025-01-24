#/usr/bin/bash

# Embryo
## Update image_directory to the path where your images are saved.
python run.py \
    --jobname='embryo' \
    --task_option='o' \
    --targetname='phaseidx' \
    --backbone_name='cnn_2D' \
    --batchsize=64 \
    --max_epoch=40 \
    --output_directory='./output' \
    --image_directory='/scratch/datasets/hk672/embryo/' \
    --image_size='224,224' \
    --csv_file_train='./demo_for_release/demo_embryo_train.csv' \
    --csv_file_val='./demo_for_release/demo_embryo_val.csv' \
    --csv_file_test='./demo_for_release/demo_embryo_test.csv' \
    --run_mode='eval' \
    --pretrained_weight

# Woundhealing
## Update image_directory to the path where your images are saved.
python run.py \
    --jobname='woundhealing' \
    --task_option='o' \
    --targetname='timepoint' \
    --backbone_name='cnn_2D' \
    --batchsize=128 \
    --max_epoch=40 \
    --output_directory='./output' \
    --image_directory='/scratch/datasets/hk672/woundhealing/data_preprocessed' \
    --image_size='224,224' \
    --csv_file_train='./demo_for_release/demo_woundhealing_train.csv' \
    --csv_file_val='./demo_for_release/demo_woundhealing_val.csv' \
    --csv_file_test='./demo_for_release/demo_woundhealing_test.csv' \
    --run_mode='eval' \
    --pretrained_weight

# Aging Brain
## Update image_directory to the path where your images are saved.
## Ensure the images have been pre-processed using the Neural Pre-Processing tool.
python run.py \
    --jobname='oasis-aging' \
    --task_option='t' \
    --targetname='age' \
    --backbone_name='cnn_3D' \
    --batchsize=16 \
    --max_epoch=100 \
    --output_directory='./output' \
    --image_directory='/share/sablab/nfs04/data/OASIS3/npp-preprocessed/image/' \
    --image_size='128,128,128' \
    --csv_file_train='./demo_for_release/demo_oasis-aging_train.csv' \
    --csv_file_val='./demo_for_release/demo_oasis-aging_val.csv' \
    --csv_file_test='./demo_for_release/demo_oasis-aging_test.csv' \
    --run_mode='eval' \
    --pretrained_weight

# MCI Brain
## Update image_directory to the path where your images are saved.
## Ensure the images have been pre-processed using the Neural Pre-Processing tool.
python run.py \
    --jobname='adni-mci' \
    --task_option='s' \
    --targetname='CDRSB' \
    --optional_meta='AGExSEX'\
    --backbone_name='cnn_3D' \
    --batchsize=16 \
    --max_epoch=40 \
    --output_directory='./output' \
    --image_directory='/scratch/datasets/hk672/adni-all-3d-preprocessed/image' \
    --image_size='128,128,128' \
    --csv_file_train='./demo_for_release/demo_adni-mci_train.csv' \
    --csv_file_val='./demo_for_release/demo_adni-mci_val.csv' \
    --csv_file_test='./demo_for_release/demo_adni-mci_test.csv' \
    --run_mode='eval' \
    --pretrained_weight

