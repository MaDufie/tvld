
import time
import os
import argparse
import torch
from torch.backends import cudnn
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.utils.utils import *
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.solver import Solver
import random
import numpy as np



def str2bool(v):
    return v.lower() in ('true')

#region: random seed
def set_seed(seed):
    seed=52
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#endregion
def run_experiment(config, dataset):
    config.dataset = dataset
    cudnn.benchmark = True

    # Create directories for saving models and results if they don't exist
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)
    
    # Create a unique results path for each divergence and dataset
    config.results_path = os.path.join(os.getcwd(), 'Results', dataset)

    if not os.path.exists(config.results_path):
        os.makedirs(config.results_path)

    solver = Solver(vars(config))

    # if config.mode == 'train':
    #     solver.train()
    if config.mode == 'train':
        start_time = time.time()

        solver.train()

        end_time = time.time()
        total_train_time = end_time - start_time
        print(f"⏱️ Total Training Time: {total_train_time:.2f} seconds")

    elif config.mode == 'test':
        # solver.test()
        start_time = time.time()

        solver.test()  # This runs your full testing pipeline

        end_time = time.time()
        inference_time = end_time - start_time
        print(f"🕒 Total Inference Time: {inference_time:.2f} seconds")

    return solver

def main(config):
    run_experiment(config, config.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=5)
    parser.add_argument('--win_size', type=int, default=20)
    parser.add_argument('--input_c', type=int, default=38)
    parser.add_argument('--output_c', type=int, default=38)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--dataset', type=str, default='SMD')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='./dataset/SMD/train.csv')
    parser.add_argument('--seed', type=int, default=52, help='Random seed')
    parser.add_argument('--model_save_path', type=str, default=os.path.join(os.getcwd(), 'checkpoints'))
    parser.add_argument('--d_model', type=int, default=702, help='Model hidden dimension size')
    parser.add_argument('--rank_threshold', type=float, default=1*1e-2, help='Threshold h1')

    config = parser.parse_args()

    args = vars(config)
    set_seed(config.seed)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')





    main(config)

