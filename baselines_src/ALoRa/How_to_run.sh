

# Instructions:
# 1. First, run the training command in the bash terminal to train the model.
# 2. After training is complete, run the testing command to evaluate the model.
# Make sure the data path is correctly set relative to your working directory.


# DATASET: SMD 
#TRAIN:
python main.py  --num_epochs 4   --batch_size 128  --mode train --dataset SMD  --data_path ./Datasets/SMD --input_c 38 --output_c 38 --win_size 20 --d_model 702
#TEST
python main.py  --num_epochs 4   --batch_size 128     --mode test    --dataset SMD   --data_path ./Datasets/SMD   --input_c 38 --win_size 20 --d_model 702 --rank_threshold 0.01

# DATASET: HAI 
#TRAIN:
python main.py   --num_epochs 4       --batch_size 256     --mode train    --dataset HAI   --data_path ./Datasets/HAI  --input_c 78    --output_c 78 --win_size 20 --d_model 3002
#TEST
python main.py   --num_epochs 4       --batch_size 128     --mode test    --dataset HAI   --data_path ./Datasets/HAI  --input_c 78    --output_c 78 --win_size 20 --d_model 3002 --rank_threshold 0.2

#DATASET: SWAT MINUTE GRANUALITY
#TRAIN:
python main.py   --num_epochs 10       --batch_size 256     --mode train    --dataset SWAT   --data_path ./Datasets/SWAT  --input_c 51    --output_c 51 --win_size 20 --d_model 512
#TEST
#better threshold
python main.py   --num_epochs 4       --batch_size 128     --mode test    --dataset SWAT   --data_path ./Datasets/SWAT  --input_c 51    --output_c 51 --win_size 20  --d_model 512 --rank_threshold 0.03


# DATASET: PSM 
# #TRAIN:
# python main.py   --num_epochs 3       --batch_size 256     --mode train    --dataset PSM   --data_path ./Datasets/PSM  --input_c 25    --output_c 25  --win_size 100 --d_model 300
# #TEST
# python main.py   --num_epochs 3       --batch_size 256     --mode test    --dataset PSM   --data_path ./Datasets/PSM  --input_c 25    --output_c 25  --win_size 100 --d_model 300 --rank_threshold 0.001


# DATASET: MSL 
#TRAIN:
python main.py  --num_epochs 3   --batch_size 256  --mode train --dataset MSL  --data_path ./Datasets/MSL --input_c 55    --output_c 55 --win_size 20 --d_model 1484 
#TEST
python main.py   --num_epochs 3      --batch_size 256     --mode test    --dataset MSL   --data_path ./Datasets/MSL  --input_c 55    --output_c 55 --win_size 20 --d_model 1484 --rank_threshold 0.03





