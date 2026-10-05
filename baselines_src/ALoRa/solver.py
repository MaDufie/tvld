import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchviz
import numpy as np
import pandas as pd
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.utils.utils import *
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.model.ALoRa import ALoRaT
from numpy.linalg import inv
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.data_factory.data_loader import get_loader_segment

from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from torch.cuda.amp import autocast, GradScaler
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.utils.affiliation.generics import convert_vector_to_events
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.utils.affiliation.metrics import pr_from_events
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.utils.evaluation.metrics import get_metrics
from Projects.AnomalyDetection.tvld.baselines_src.ALoRa.utils.evaluation.metrics import run_tapr



def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.dataset = dataset_name
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
        print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        if self.counter >= self.patience:
            self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        filename = f'{self.dataset}_checkpoint.pth'
        torch.save(model.state_dict(), os.path.join(path, filename))
        #region: Save with datetime:
                # # Generate date string
        # date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        # # Create filename with dataset name and date
        # filename = f'{self.dataset}_checkpoint_{date_str}.pth'
        # # Save the model
        # torch.save(model.state_dict(), os.path.join(path, filename))
        # Save the model with a fixed name (overwrite every time)
        #endregion
        self.val_loss_min = val_loss


class Solver(object):
    DEFAULTS = {}
    def __init__(self, config):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        self.results_path = config['results_path']  # Add this line

        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("CUDA not available. Using CPU.")

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                               mode='train',
                                               dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,
                                              mode='val',
                                              dataset=self.dataset)
        self.test_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,step=1,
                                              mode='test',
                                              dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.data_path, batch_size=self.batch_size, win_size=self.win_size,step=1,
                                              mode='thre',
                                              dataset=self.dataset)

        self.build_model()
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print(f'device {self.device}')
        print(torch.cuda.is_available())  
        print(torch.cuda.current_device())  
        print(torch.cuda.get_device_name(0))  

        self.criterion = nn.MSELoss()
                   
    def build_model(self):
        self.model = ALoRaT(win_size=self.win_size, d_model=self.d_model,enc_in=self.input_c, c_out=self.output_c, e_layers=3,dataset=self.dataset)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device) # Move the model to the correct device

    def vali(self, vali_loader):
        self.model.eval()
        torch.cuda.empty_cache() # remove uneccesary memory
        RecLoss = []

        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, SA , _ = self.model(input)
            rec_loss = self.criterion(output, input)
            RecLoss.append((rec_loss).item())
        return np.average(RecLoss)
    
    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        # path = self.model_save_path
        path = os.path.join(self.model_save_path, self.dataset)                  #original
        # path = os.path.join(self.model_save_path, self.dataset,'seed52/T20') #  for ablitation study of WindowSize
        #region: Count total trainable parameters
        def count_parameters(model: torch.nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        total_params = count_parameters(self.model)
        print(f"[INFO] Total Trainable Parameters: {total_params:,}")
        #endregion:
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=self.dataset)

        train_steps = len(self.train_loader)
        self.rank_data_over_epochs = []  # Reset for each training session
        # # Store 2nd to 5th largest singular values across all batches
        # singular_distributions = {1: [], 2: [], 3: [], 4: []}  # 1=2nd, 2=3rd, 3=4th, 4=5th

        for epoch in range(self.num_epochs):
            print(f' \n =============================== Epoch {epoch} ===============================')
            iter_count = 0
            Tloss_list = []
            epoch_ranks = []  # To store rank per sample in the current epoch
            recloss_list = []
            ALoRaReg_list = []
            epoch_time = time.time()

            total_embed_time = 0.0

            self.model.train() # training mode

            for i, (input_data, labels) in enumerate(self.train_loader):
                print(f'==== Batch {i} ====')
                self.optimizer.zero_grad()
                # ---- Start total training timer ----

                iter_count += 1
                input = input_data.float().to(self.device)
                output, SA, Emb_data = self.model(input)
                batch_start = time.time()

              
                if epoch == self.num_epochs - 1:
                    final_SA_last_layer = SA[-1]  # Last layer attention: [B, H, T, T]
                    avg_attention = torch.mean(final_SA_last_layer, dim=1)  # Avg over heads: [B, T, T]
                    
                    for b in range(avg_attention.shape[0]):
                        attention_matrix = avg_attention[b]
                        singular_values = torch.linalg.svd(attention_matrix, full_matrices=False)[1].detach().cpu().numpy()


                ALoRaLoss = 0.0
                batch_ranks = []  # Store ranks per layer for the entire batch
                
                def calculate_low_rank_loss(SA_avg, reg_type, **kwargs):
                    singular_values = torch.linalg.svd(SA_avg, full_matrices=False)[1]

                    if reg_type == "nuclear_norm":
                        # Nuclear norm: \sum_{i=1}^k \lambda \sigma_i
                        low_rank_loss = torch.sum(kwargs.get("lambda", 1.0) * singular_values)

                    elif reg_type == "tnn":
                        # TNN: \sum_{i=r+1}^k \lambda \sigma_i
                        r = kwargs.get("r", 0)
                        low_rank_loss = torch.sum(kwargs.get("lambda", 1.0) * singular_values[r:])
                    elif reg_type == "geman":
                        # Geman: \sum_{i=1}^k \lambda \frac{\sigma_i}{\sigma_i+\gamma}
                        gamma = kwargs.get("gamma", 1.0)
                        low_rank_loss = torch.sum(kwargs.get("lambda", 1.0) * singular_values / (singular_values + gamma))

                    elif reg_type == "tnn_geman":
                        # TNN + Geman: Apply Geman penalty after skipping the first r singular values
                        r = kwargs.get("r", 1)
                        gamma = kwargs.get("gamma", 1.0)
                        truncated_singular_values = singular_values[r:]
                        low_rank_loss = torch.sum(kwargs.get("lambda", 1.0) * truncated_singular_values / (truncated_singular_values + gamma))

                    else:
                        raise ValueError(f"Unknown regularization type: {reg_type}")

                    return low_rank_loss


                # Optimization:
                for u in range(len(SA)):  # Loop through layers      
                    SA_avg = torch.mean(SA[u], dim=1) # Average across heads to shape [batch, timesteps, timesteps]
                    
                    # Calculate singular values for rank penalty
                    singular_values = torch.linalg.svd(SA_avg, full_matrices=False)[1]
                    low_rank_loss = calculate_low_rank_loss(SA_avg, reg_type="tnn_geman", r=1,gamma=1.0)
                    # Log rank or singular values for debugging
                    ranks = torch.sum(singular_values > 1e-4, dim=1).cpu().numpy()
                    batch_ranks.append(ranks)
                    ALoRaLoss_l = low_rank_loss 

                    # Accumulate the layer low rank loss into the total SA loss
                    ALoRaLoss += ALoRaLoss_l

                    # Log values for debugging
                    print(f"Layer {u+1}  | Low-Rank Loss: {ALoRaLoss_l.item()}")

                
                ## Average ALoRaLoss over all layers
                ALoRaLoss = ALoRaLoss / len(SA)

                # After processing all layers, transpose batch_ranks to shape [num_samples, num_layers]
                # batch_ranks = np.array(batch_ranks).T  # Shape to [num_samples_in_batch, num_layers]
                
                # epoch_ranks.append(batch_ranks)  # Append ranks for this batch                
                #region Compute reconstruction loss
                rec_loss = self.criterion(output, input)
                lambda_reg = 10 # 
                # Combine reconstruction loss with regularization
                Tloss = rec_loss + lambda_reg*ALoRaLoss 
                Tloss_list.append(Tloss.item())
                recloss_list.append(rec_loss.item())
                #endregion
                ALoRaReg_list.append(ALoRaLoss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # Optimization steps : 
                self.optimizer.zero_grad()
                Tloss.backward(retain_graph=False)
                self.optimizer.step()

            # After processing all batches in the epoch, convert epoch_ranks to shape [num_samples, num_layers]
            # epoch_ranks = np.concatenate(epoch_ranks, axis=0)  # Combine batches into a single array for the epoch
            # self.rank_data_over_epochs.append(epoch_ranks)
    
            # Average rank data for the last layer and store it for this epoch
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(Tloss_list)
            Reg_loss = np.average(ALoRaReg_list)
            rec_loss = np.average(recloss_list)
            vali_loss = self.vali(self.test_loader)

            print(
                "Train Loss: {0:.7f}   rec_loss: {1:.7f} Reqularization_loss: {2:.7f} ".format(train_loss,rec_loss,Reg_loss))

            early_stopping(vali_loss,self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)
        # print(f" Total embedding time: {total_embed_time:.2f} seconds")

        # === Plot subplots for 2nd to 5th largest singular values ===
        # fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        # axes = axes.flatten()
        # bins = np.logspace(-4, 0, 40)

        # titles = {
        #     1: '4th Largest',
        #     2: '5th Largest',
        #     3: '4th Largest',
        #     4: '5th Largest'
        # }

        # for i, ax in enumerate(axes, start=1):  # i = 1 to 4
        #     data = singular_distributions[i]
        #     ax.hist(data, bins=bins, edgecolor='black', alpha=0.75)
        #     ax.set_xscale('log')
        #     ax.set_title(f'{titles[i]} Singular Value')
        #     ax.set_xlabel('Singular Value (log scale)')
        #     ax.set_ylabel('Frequency')
        #     ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # plt.tight_layout()
        # save_path_multi = os.path.join(self.results_path, 'singular_value_distributions_2nd_to_5th_subplots.png')
        # plt.savefig(save_path_multi)
        # print(f"Saved 2nd to 5th singular value distribution subplots at: {save_path_multi}")
        # plt.close()

        # # Create final checkpoint filename with timestamp
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # final_checkpoint_filename = f"{self.dataset}_checkpoint_{timestamp}.pth"
        # final_checkpoint_path = os.path.join(self.model_save_path, final_checkpoint_filename)

        # # Save the model
        # torch.save(self.model.state_dict(), final_checkpoint_path)
        # print(f"Final model checkpoint saved at: {final_checkpoint_path}")

        
        
###### Testing phase  #########
    def test(self):
        
        ## Load the model checkpoint for the current divergence type
        model_path = os.path.join(self.model_save_path, self.dataset ,f'{self.dataset}_checkpoint.pth')
        ## Define the full dataset-specific model path
        print(f"Loading checkpoint from: {model_path}")
        self.model.load_state_dict(torch.load(model_path))
    
        self.model.eval() # sets the model to evaluation mode, affecting layers like dropout and batch normalization which behave differently during training vs. testing.dropout is disabled and batch normalization uses the population statistics instead of the batch statistics.
        print("======================TEST MODE======================")

        # Add this block to inspect memory:
        print("======================TEST MODE======================")
        # print('GPU Info:')
        # print('Total memory:', torch.cuda.get_device_properties(self.device).total_memory / 1024**3, 'GB')
        # print('Allocated memory:', torch.cuda.memory_allocated(self.device) / 1024**2, 'MB')
        # print('Reserved memory:', torch.cuda.memory_reserved(self.device) / 1024**2, 'MB')


        criterion = nn.MSELoss(reduction = 'none')
        criterionLoc = lambda x, y: (x - y) ** 2
        test_labels = []
        AScore = []
        output_data_list = []
        input_data_list = []
        rec_loss= []  # To store metric values for each time step
        
        # For saving attention matrix ranks across layers
        ranks_over_layers = []
        ranks_over_layers_unth = []
        Window_labels = []  # To store whether each sample's window is anomalous or normal

        total_samples = 0
        start_test_time = time.time()
        total_embed_time = 0
        for i, (input_data, labels) in enumerate(self.thre_loader):
            total_samples += input_data.shape[0]

            batch_ranks = []  # Store ranks per layer for the entire batch
            batch_ranks_unth = []
            input = input_data.float().to(self.device)
            batch_start = time.time()

            # # ---- Time embedding separately ----
            embed_start = time.time()
            Emb_data = self.model.embedding(input)
            total_embed_time += time.time() - embed_start

            output, SA, Emb_data = self.model(input)
            # Compute the loss
            # For the first batch (i == 0), process the first window differently
            if i == 0:
                # Handle the first window in the first batch
                first_window_input = input[0:1, :, :]  # First window in the batch (shape: [1, time_steps, features])
                # print(f"first_window_input shape: {first_window_input.shape}")

                first_window_output = output[0:1, :, :]  # Corresponding output for the first window
                # print(f"first_window_output shape: {first_window_output.shape}")

                loss = torch.mean(criterion(first_window_input, first_window_output), dim=-1).to(self.device)
                # print(f"loss shape: {loss.shape}")
                cri = loss.detach().cpu().numpy()  # Move to CPU and convert to NumPy
                cri = cri.reshape(-1)  # Shape becomes (100,)
                # print(f"First batch, first window: Using full window loss, cri shape: {cri.shape}")
                # Add corresponding labels for all time steps in the first window
                test_labels.extend(labels[0, :].cpu().numpy().tolist())  # Shape: (time_steps,)

                # Handle the remaining windows in the first batch
                other_windows_input = input[1:, -1, :]  # Last timestep of remaining windows in the batch
                other_windows_output = output[1:, -1, :]
                loss = torch.mean(criterion(other_windows_input, other_windows_output), dim=-1).to(self.device)
                cri_remaining = loss.detach().cpu().numpy()
                # Append both criteria to attens_energy
                AScore.append(cri)
                AScore.append(cri_remaining)
                attens_energy_array = np.concatenate(AScore, axis=0)
                print(f'{i}|{attens_energy_array.shape}')
                 # Add corresponding labels for the last time step of remaining windows
                test_labels.extend(labels[1:, -1].cpu().numpy().tolist())  # Shape: (batch_size-1,)
                       
            else:
                # For all subsequent batches, use last timestep loss for all windows
                last_timestep_input = input[:, -1, :]  # Last timestep of each window in the batch
                last_timestep_output = output[:, -1, :]
                loss = torch.mean(criterion(last_timestep_input, last_timestep_output), dim=-1).to(self.device)
                cri = loss.detach().cpu().numpy()
                # print(f"Batch {i}: Using last timestep loss, cri shape: {cri.shape}")
                AScore.append(cri)
                 # Add corresponding labels for the last time step of all windows
                test_labels.extend(labels[:, -1].cpu().numpy().tolist())  # Shape: (batch_size,)

            
            # Convert labels to numpy to facilitate window anomaly calculation
            label_np = labels.cpu().numpy()

        
            for u in range(len(SA)):
                # Average across heads to shape [batch, timesteps, timesteps]
                series_avg = torch.mean(SA[u], dim=1)
                 # Calculate rank per sample in the batch
                singular_values = torch.linalg.svd(series_avg, full_matrices=False)[1]
                rank_thre = self.rank_threshold
                ranks = torch.sum(singular_values > rank_thre, dim=1).cpu().numpy() # MY GENRATED DATA :1e-2, smd:
                if i == 0:
                    # Repeat the first value 99 more times at the beginning
                    # repeated_ranks = np.concatenate([[ranks[0]] * 99, ranks])  # FOR T=100
                    repeated_ranks = np.concatenate([[ranks[0]] * 19, ranks])    # FOR T=20
                    # print(f"First batch, adjusted ranks shape: {repeated_ranks.shape}")
                    batch_ranks.append(repeated_ranks)
                else:
                    # Append the regular ranks
                    batch_ranks.append(ranks)
    
            batch_ranks = np.array(batch_ranks).T  # Shape to [num_samples_in_batch, num_layers]
            ranks_over_layers.append(batch_ranks)
            rank_metric = np.mean(batch_ranks, axis=1)  # Average ranks across layers

            rec_loss.append(loss.detach().cpu().numpy())
        
        end_test_time = time.time()
        inference_duration = end_test_time - start_test_time
        per_sample_time = inference_duration / total_samples
        embed_time_per_sample = total_embed_time / total_samples

        print(f'Per sample inferecne time: {per_sample_time}')
        # print(f'Per sample emb time: {embed_time_per_sample}')
        # Concatenate ranks across batches
        ranks_over_layers = np.concatenate(ranks_over_layers, axis=0)  # Shape: [num_samples, num_layers]
        
        ALoRaT_score = ranks_over_layers[:,-1]  # At the Last layer
        print(f'the shape of rank loss is {ALoRaT_score.shape}')
        
        indices = np.arange(len(ALoRaT_score))  # Generate indices corresponding to the ALoRaT_score values
        test_labels = np.array(test_labels)

        
      #region PYPLOTS-SCORES Define save path
        # save_path_html = os.path.join(self.results_path, f"rank_loss_over_ascore.html")

        # # Create a figure
        # fig = go.Figure()

        # # Add the anomaly score line
        # fig.add_trace(go.Scatter(
        #     x=indices,
        #     y=ALoRaT_score,
        #     mode='lines+markers',
        #     line=dict(color='blue', width=2),
        #     name='Rank Loss',
        #     marker=dict(size=4)
        # ))

        # # Identify and highlight anomalous regions
        # in_anomaly = False
        # start_idx = 0
        # for i in range(len(test_labels)):
        #     if test_labels[i] == 1 and not in_anomaly:
        #         start_idx = i
        #         in_anomaly = True
        #     elif test_labels[i] == 0 and in_anomaly:
        #         end_idx = i
        #         fig.add_vrect(
        #             x0=indices[start_idx], x1=indices[end_idx - 1],
        #             fillcolor='lightcoral', opacity=0.3, line_width=0,
        #             layer='below', name="Anomaly Region"
        #         )
        #         in_anomaly = False

        # # Edge case: If it ends with an anomaly
        # if in_anomaly:
        #     fig.add_vrect(
        #         x0=indices[start_idx], x1=indices[-1],
        #         fillcolor='lightgreen', opacity=0.3, line_width=0,
        #         layer='below', name="Anomaly Region"
        #     )

        # # Update layout
        # fig.update_layout(
        #     title=f"Rank Loss Plot",
        #     xaxis_title='Index',
        #     yaxis_title='Rank Loss',
        #     hovermode='x unified',
        #     plot_bgcolor='white',
        #     template='simple_white',
        #     font=dict(size=14)
        # )

        # # Save and show
        # fig.write_html(save_path_html)
        # fig.show()

        # print(f"rank Plot saved at: {save_path_html}")


        #endregion
        #region : Static Plot
        # Example values (in your real code, you already have these)
        # ALoRaT_score = np.array([...])
        # test_labels = np.array([...])
        # indices = np.arange(len(ALoRaT_score))

        # # === Save path
        # save_path1 = os.path.join(self.results_path, f"rank_loss_over_ascore.png")

        # # === Plot
        # plt.figure(figsize=(12, 6))

        # # --- Highlight anomalous regions in light coral ---
        # in_anomaly = False
        # start_idx = 0

        # for i in range(len(test_labels)):
        #     if test_labels[i] == 1 and not in_anomaly:
        #         start_idx = i
        #         in_anomaly = True
        #     elif test_labels[i] == 0 and in_anomaly:
        #         end_idx = i
        #         plt.axvspan(indices[start_idx], indices[end_idx - 1], color='lightcoral', alpha=0.3)
        #         in_anomaly = False

        # # Edge case: if last segment is anomaly
        # if in_anomaly:
        #     plt.axvspan(indices[start_idx], indices[-1], color='lightcoral', alpha=0.3)

        # # --- Plot the score line in black ---
        # plt.plot(indices, ALoRaT_score, color='black', linewidth=1.5, label='Rank Loss')

        # # === Plot config
        # plt.xlabel('Index', fontsize=12)
        # plt.ylabel('Rank Loss', fontsize=12)
        # plt.title(f'Rank Loss Over Indices', fontsize=14)
        # plt.grid(alpha=0.3)
        # plt.tight_layout()

        # # Save
        # plt.savefig(save_path1, dpi=300)
        # plt.close()

        # print(f"Rank loss plot saved at: {save_path1}")

        #endregion

        #region Plot saving -Previous one 
        # Assuming ALoRaT_score and test_labels are numpy arrays and indices are already defined
        # save_path1 = os.path.join(self.results_path, f"rank_loss_over_ascore.png")

        # # Start the plot
        # plt.figure(figsize=(10, 6))

        # # Plot segments based on test_labels
        # # for i in range(300):
        # for i in range(len(ALoRaT_score) - 1):

        #     if test_labels[i] == 1:  # Anomalous region
        #         plt.plot(indices[i:i + 2], ALoRaT_score[i:i + 2], color='red',marker='o')
        #     else:  # Normal region
        #         plt.plot(indices[i:i + 2], ALoRaT_score[i:i + 2], color='blue',marker='.')

        # # Add labels, title, and legend
        # plt.xlabel('Index', fontsize=12)
        # plt.ylabel('Rank Loss', fontsize=12)
        # plt.title('Rank Loss Over Indices', fontsize=14)
        # plt.grid(alpha=0.3)
        # plt.tight_layout()

        # # Save and close the plot
        # plt.savefig(save_path1, dpi=300)
        # plt.close()

        # print(f"Plot saved successfully at {save_path1}")
        #endregion 
        #region:scoring
        print(f"Shape of ranks_over_layers: {ranks_over_layers.shape}")
        print(f"Shape of ALoRaT_score: {ALoRaT_score.shape}")
        Window_labels = np.array(Window_labels)  # Convert to numpy array
        AScore_t = np.concatenate(AScore, axis=0).reshape(-1)
        print(f'the shape of AScore_t is {AScore_t.shape}')
        
        rec_loss = np.concatenate(rec_loss, axis=0).reshape(-1)

        # test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        AScore = np.array((AScore_t)*ALoRaT_score)
        #endregion

        
        # Plot rank trends across layers (no epoch dimension in testing)
        # self.plot_layerwise_rank_trends(ranks_over_layers)
        # self.plot_layerwise_rank_trends(ranks_over_layers, Window_labels)
        # self.plot_layerwise_ranks_per_window(ranks_over_layers, Window_labels)
        #region PLOTY OF FINAL ANOMALY DETECTION SCORE:
        # indices = np.arange(len(test_energy1))  # x-axis
        # save_path_html = os.path.join(self.results_path, f"final_score_plot.html")

        # # === Create figure ===
        # fig = go.Figure()

        # # === Add the final anomaly score (test_energy1) line ===
        # fig.add_trace(go.Scatter(
        #     x=indices,
        #     y=test_energy1,
        #     mode='lines+markers',
        #     line=dict(color='blue', width=2),
        #     name='Final Anomaly Score (test_energy1)',
        #     marker=dict(size=4)
        # ))

        # # === Highlight anomaly regions with soft green background ===
        # in_anomaly = False
        # start_idx = 0
        # for i in range(len(test_labels)):
        #     if test_labels[i] == 1 and not in_anomaly:
        #         start_idx = i
        #         in_anomaly = True
        #     elif test_labels[i] == 0 and in_anomaly:
        #         end_idx = i
        #         fig.add_vrect(
        #             x0=indices[start_idx], x1=indices[end_idx - 1],
        #             fillcolor='lightcoral', opacity=0.3, line_width=0,
        #             layer='below', name="Anomaly Region"
        #         )
        #         in_anomaly = False

        # # Edge case: if SA ends in anomaly
        # if in_anomaly:
        #     fig.add_vrect(
        #         x0=indices[start_idx], x1=indices[-1],
        #         fillcolor='lightgreen', opacity=0.3, line_width=0,
        #         layer='below', name="Anomaly Region"
        #     )

        # # === Final layout tweaks ===
        # fig.update_layout(
        #     title=f"Final Anomaly Score Plot",
        #     xaxis_title='Index',
        #     yaxis_title='Final Anomaly Score',
        #     hovermode='x unified',
        #     plot_bgcolor='white',
        #     template='simple_white',
        #     font=dict(size=14)
        # )

        # # === Save and show ===
        # fig.write_html(save_path_html)
        # fig.show()

        # print(f" Interactive plot saved at: {save_path_html}")
    
        #endregion


        
        # save_path2 = os.path.join(self.results_path, "RecLoss_overindices.png")

        # Start the plot
        # plt.figure(figsize=(10, 6))

        # for i in range(len(AScore_t) - 1):

        #     if test_labels[i] == 1:  # Anomalous region
        #         plt.plot(indices[i:i + 2], AScore_t[i:i + 2], color='red',marker='o')
        #     else:  # Normal region
        #         plt.plot(indices[i:i + 2], AScore_t[i:i + 2], color='blue',marker='o')

        # # Add labels, title, and legend
        # plt.xlabel('Index', fontsize=12)
        # plt.ylabel('Rec Loss', fontsize=12)
        # plt.title('Reconstruction loss Over Indices with Anomalies Highlighted', fontsize=14)
        # plt.grid(alpha=0.3)
        # plt.tight_layout()

        # # Save and close the plot
        # plt.savefig(save_path2, dpi=300)
        # plt.close()

        # print(f"Plot saved successfully at {save_path2}")
        
        
        ##region === Save TokenEmbedding Contribution Matrix ===
        # if hasattr(self.model.embedding, "value_embedding"):
        #     token_embed = self.model.embedding.value_embedding
        #     if hasattr(token_embed, "save_weight_contribution_matrix"):
        #         save_path = os.path.join('./Localization/PATHS', self.dataset ,f"filter_weight_matrix_{self.dataset}.npy")
        #         token_embed.save_weight_contribution_matrix(save_path)
        ##endregion
        
        gt = test_labels.astype(int)
        anomaly_ratio = np.sum(gt == 1) / len(gt)
        # print(f'anomaly ratio is :{anomaly_ratio*100}')
        # Calculate the 80th to 100th percentile thresholds
        percentile_range = np.linspace(70, 100, 200)  # 100 steps
        thresholds = np.percentile(AScore, percentile_range)

        def evaluate_energy(test_energy, thresholds, test_labels, percentile_range):
            best_threshold = None
            best_f1_score = 0
            best_metrics = None
            best_pred = None
            best_percentile = None

            for i, thresh in enumerate(thresholds):
                pred = (test_energy > thresh).astype(int)

                events_pred = convert_vector_to_events(pred)
                events_gt = convert_vector_to_events(test_labels)
                Trange = (0, len(test_labels))

                event_metrics = pr_from_events(events_pred, events_gt, Trange)
                precision = event_metrics['precision']
                recall = event_metrics['recall']
                f1_score = 2 * precision * recall / (precision + recall + 1e-8)

                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_threshold = thresh
                    best_metrics = (precision, recall, f1_score)
                    best_pred = pred
                    best_percentile = percentile_range[i]

            print(f"Best threshold: {best_threshold:.6f}, Percentile used: {best_percentile}th")

            return best_threshold, best_metrics, best_pred
        threshold, metrics, pred = evaluate_energy(AScore, thresholds, gt,percentile_range)
        pred = (AScore > threshold).astype(int)
        # Define the file path for saving performance results
        performance_file = os.path.join(self.results_path, f"performance.txt")

            
        print(f'==== ALoRa-Det Perfmormance for dataset{self.dataset}')            
        events_pred = convert_vector_to_events(pred)
        events_gt = convert_vector_to_events(test_labels)
        Trange = (0, len(test_labels))  # Time range of the predictions
        event_metrics = pr_from_events(events_pred, events_gt, Trange)

        precision = event_metrics['precision']
        recall = event_metrics['recall']
        f1_score = 2 * precision * recall / (precision + recall)
        print(f"Event-Based Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")
        print('============================================')            
    
        
        with open(performance_file, "a") as f:
            f.write("\n =============================== EVENT-BASED Performance =================================================\n")
            f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}\n\n")
            print(f"Performance results saved successfully to {performance_file}")
        # metrics = get_metrics(AScore, test_labels.flatten(), slidingWindow=50, pred=pred)

        # for k, v in metrics.items():
        #     print(f"{k}: {v:.4f}")     


        # for theta in [0.3,0.5,0.8]:
        #     for delta in [10,20, 50]:
        #         result = run_tapr(pred, gt, theta=theta, delta=delta, verbose=True)
                        # with open(performance_file, "a") as f:
        #     f.write("\n =============================== Time-Series Aware Metrics (TaPR) =======================================\n")
        #     f.write(f"TaR: {tapr_metrics['TaR']:.4f} (Detection: {tapr_metrics['TaR_detection']:.4f}, Portion: {tapr_metrics['TaR_portion']:.4f})\n")
        #     f.write(f"TaP: {tapr_metrics['TaP']:.4f} (Detection: {tapr_metrics['TaP_detection']:.4f}, Portion: {tapr_metrics['TaP_portion']:.4f})\n\n")   

        # tapr_metrics = run_tapr(pred, test_labels, theta=0.5, delta=50, labels=(0, 1), verbose=True)

        # # Optionally log or save:
        # with open(performance_file, "a") as f:
        #     f.write("\n =============================== Time-Series Aware Metrics (TaPR) =======================================\n")
        #     f.write(f"TaR: {tapr_metrics['TaR']:.4f} (Detection: {tapr_metrics['TaR_detection']:.4f}, Portion: {tapr_metrics['TaR_portion']:.4f})\n")
        #     f.write(f"TaP: {tapr_metrics['TaP']:.4f} (Detection: {tapr_metrics['TaP_detection']:.4f}, Portion: {tapr_metrics['TaP_portion']:.4f})\n\n")   
############################################### End of ALoRa ##########################################################################


# region Loclaiziton save the rec loss:
    # def test(self):
    #     import numpy as np
    #     import torch
    #     import os
    #     import matplotlib.pyplot as plt

    #     # === Config Paths ===
    #     results_path = os.path.join('Results', self.dataset)
    #     os.makedirs(results_path, exist_ok=True)

    #     ##Load the model checkpoint
    #     model_path = os.path.join(self.model_save_path, self.dataset,f'{self.dataset}_checkpoint.pth')

    #     print(f"Loading checkpoint from: {model_path}")
    #     self.model.load_state_dict(torch.load(model_path))
    #     self.model.load_state_dict(torch.load(model_path))
    #     self.model.eval()

    #     print("======================TEST MODE======================")

    #     criterion = nn.MSELoss(reduction='none')
    #     # criterionLoc = lambda x, y: (x ranks- y) ** 2  # Per-feature localization
    #     criterionLoc = lambda x, y: (x - y) ** 2

    #     # === Step 1: Threshold stats ===
    #     attens_energy = []
    #     for input_data, labels in self.train_loader:
    #         input = input_data.float().to(self.device)
    #         output, _, _ = self.model(input)
    #         loss = torch.mean(criterion(input, output), dim=-1)
    #         attens_energy.append(loss.detach().cpu().numpy())
    #     train_energy = np.concatenate(attens_energy).reshape(-1)

    #     # # === Step 2: Compute test loss for threshold ===
    #     # attens_energy = []
    #     # for input_data, labels in self.thre_loader:
    #     #     input = input_data.float().to(self.device)
    #     #     output, _, _= self.model(input)
    #     #     loss = torch.mean(criterion(input, output), dim=-1)
    #     #     attens_energy.append(loss.detach().cpu().numpy())
    #     # test_energy = np.concatenate(attens_energy).reshape(-1)

    #     # combined_energy = np.concatenate([train_energy, test_energy])
    #     # thresh = np.percentile(combined_energy, 100 - self.anormly_ratio)
    #     # print(" Threshold:", thresh)

    #     # === Step 3: Main RecLoc Loop ===
    #     test_labels = []
    #     recloc_windows = []

    #     for i, (input_data, labels) in enumerate(self.thre_loader):
    #         input = input_data.float().to(self.device)
    #         output, _, _= self.model(input)

    #         if i == 0:
    #             # First window
    #             first_input = input[0:1]        # [1, T, D]
    #             first_output = output[0:1]
    #             rec_full = criterionLoc(first_input, first_output)[0]  # [T, D]
    #             recloc_windows.append(rec_full.detach().cpu().numpy())
    #             test_labels.extend(labels[0, :].cpu().numpy().tolist())

    #             # Remaining windows in first batch
    #             other_input = input[1:, -1, :]  # [B-1, D]
    #             other_output = output[1:, -1, :]
    #             rec_last = criterionLoc(other_input, other_output)
    #             for vec in rec_last:
    #                 recloc_windows.append(vec.detach().unsqueeze(0).cpu().numpy())
    #             test_labels.extend(labels[1:, -1].cpu().numpy().tolist())
    #         else:
    #             # Later batches
    #             last_input = input[:, -1, :]
    #             last_output = output[:, -1, :]
    #             rec_last = criterionLoc(last_input, last_output)
    #             for vec in rec_last:
    #                 recloc_windows.append(vec.detach().unsqueeze(0).cpu().numpy())
    #             test_labels.extend(labels[:, -1].cpu().numpy().tolist())

    #     # === Save RecLoc Matrix ===
    #     RecLoc = np.concatenate(recloc_windows, axis=0)
    #     save_dir = f"./LocaliationFiles/{self.dataset}"
    #     os.makedirs(save_dir, exist_ok=True)

    #     print(f"RecLoc shape: {RecLoc.shape} (timesteps × features)")
    #     np.save(os.path.join(save_dir, "RecLoc_matrix.npy"), RecLoc)
# endregion


