#!/bin/bash

num_epochs=10

echo "exxact@1" | sudo -S  $(which python) EEG_level_head.py \
        --mode train \
        --dataset NORMAL \
        --train_csv_dirs "/data/NORMAL/EEG_level/continuous_results" \
        --file_list_path /data/NORMAL/EEG_level/training_list.csv \
        --output_dir checkpoints/EEG_level_NORMAL6  \
        --num_epochs ${num_epochs} \
        --pe_max_length 15000 \
        --lr 5e-4 \
        --save_freq 1 \
        --resume_training


SLOWING_datasets=("FOC_SLOWING" "GEN_SLOWING")
for SLOWING_dataset in "${SLOWING_datasets[@]}"; do
  echo "exxact@1" | sudo -S  $(which python) EEG_level_head.py \
          --mode train \
          --dataset ${SLOWING_dataset} \
          --train_csv_dirs "/data/SLOWING/EEG_level/continuous_results" \
          --file_list_path /data/SLOWING/EEG_level/training_list.csv \
          --output_dir checkpoints/EEG_level_${SLOWING_dataset}6  \
          --num_epochs ${num_epochs} \
          --pe_max_length 15000 \
          --focal_alpha "0.25" \
          --lr 5e-4 \
          --save_freq 1 \
          --resume_training
done


echo "exxact@1" | sudo -S  $(which python) EEG_level_head.py \
        --mode train \
        --dataset BS \
        --train_csv_dirs "/data/BS/EEG_level/continuous_results" \
        --file_list_path /data/BS/EEG_level/training_list.csv \
        --output_dir checkpoints/EEG_level_BS6  \
        --num_epochs ${num_epochs} \
        --pe_max_length 15000 \
        --focal_alpha "0.25" \
        --lr 5e-4 \
        --save_freq 1 \
        --resume_training


echo "exxact@1" | sudo -S  $(which python) EEG_level_head.py \
        --mode train \
        --dataset SPIKES \
        --train_csv_dirs "/data/SPIKES/EEG_level/continuous_results" \
        --file_list_path /data/SPIKES/EEG_level/training_list.csv \
        --output_dir checkpoints/EEG_level_SPIKES6  \
        --num_epochs ${num_epochs} \
        --pe_max_length 15000 \
        --lr 5e-4 \
        --save_freq 1 \
        --resume_training


FOC_GEN_SPIKES_datasets=("FOC_SPIKES" "GEN_SPIKES")
for FOC_GEN_SPIKES_dataset in "${FOC_GEN_SPIKES_datasets[@]}"; do
  echo "exxact@1" | sudo -S  $(which python) EEG_level_head.py \
          --mode train \
          --dataset ${FOC_GEN_SPIKES_dataset} \
          --train_csv_dirs "/data/FOC_GEN_SPIKES/EEG_level/continuous_results" \
          --file_list_path /data/FOC_GEN_SPIKES/EEG_level/training_list.csv \
          --output_dir checkpoints/EEG_level_${FOC_GEN_SPIKES_dataset}6 \
          --num_epochs ${num_epochs} \
          --pe_max_length 15000 \
          --focal_alpha "0.25" \
          --lr 5e-4 \
          --save_freq 1 \
          --resume_training
done



IIIC_datasets=("SEIZURE" "LPD" "GPD" "LRDA" "GRDA")
for IIIC_dataset in "${IIIC_datasets[@]}"; do
  echo "exxact@1" | sudo -S  $(which python) EEG_level_head.py \
          --mode train \
          --dataset ${IIIC_dataset} \
          --train_csv_dirs "/data/IIIC/EEG_level/continuous_results" \
          --file_list_path /data/IIIC/EEG_level/training_list.csv \
          --output_dir checkpoints/EEG_level_${IIIC_dataset}6  \
          --num_epochs ${num_epochs} \
          --pe_max_length 15000 \
          --focal_alpha "0.25" \
          --lr 5e-4 \
          --save_freq 1 \
          --resume_training
done



