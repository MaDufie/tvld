#SMD
python ./data_factory/Preprocess/Preprocess.py \
  --dataset SMD \
  --file ./Datasets/SMD/train.csv

# msl
python ./data_factory/Preprocess/Preprocess.py \
  --dataset MSL \
  --file ./Datasets/MSL/MSL_train.npy

#PSM
python ./data_factory/Preprocess/Preprocess.py \
  --dataset PSM \
  --file ./Datasets/PSM/train.csv \
  --drop_first_col

# HAI
python ./data_factory/Preprocess/Preprocess.py \
  --dataset HAI \
  --file ./Datasets/HAI/train.csv \
  --drop_first_col

#swat
python ./data_factory/Preprocess/Preprocess.py \
  --dataset SWAT \
  --file ./checkpoints/SWAT/train.csv \
  --drop_first_col
