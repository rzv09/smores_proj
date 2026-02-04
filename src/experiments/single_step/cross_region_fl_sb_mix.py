import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

import data_processing.three_oec.process_3oec as process_3oec
import data_processing.three_oec.sequences_3oec as sequences_3oec 
import data_processing.santa_barbara.process_sb as process_sb
from data_processing.data_utils import standardize_piece, create_sequences_single_step, make_sequence_tensor 
import models.lstm.single_step as lstm_single_step
import plotting.plot_preds
import utils
import utils.set_seeds
from metrics.np.regression import MARE
from utils.write_metrics import write_csv, write_stats_mare, write_stats_crit
from plotting.plot_mare import plot_MARE
from plotting.plot_smoothl1loss import plot_smooth_l1_loss

class CrossRegionFL4SB12():
    def __init__(self, data_path_train, data_path_test, num_train_epochs, sampling_freq, sequence_len, device):
        self.data_path = data_path_train
        self.data_path_test = data_path_test
        self.num_train_epochs = num_train_epochs
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device

    def process_data(self):
        # preprocess SB data (train + test)
        self.df_sb = process_sb.make_timeseries(self.data_path)

        self.sb1 = process_sb.get_first_piece(self.df_sb)
        self.sb1_mean = self.sb1.mean().values
        self.sb1_std = self.sb1.std().values
        self.sb2 = process_sb.get_second_piece(self.df_sb)
        self.sb2_mean = self.sb2.mean().values
        self.sb2_std = self.sb2.std().values
        self.sb3 = process_sb.get_third_piece(self.df_sb)
        self.sb3_mean = self.sb3.mean().values
        self.sb3_std = self.sb3.std().values
        self.sb4 = process_sb.get_fourth_piece(self.df_sb)
        self.sb4_mean = self.sb4.mean().values
        self.sb4_std = self.sb4.std().values

        # preprocess FL data (train)
        self.df_fl = process_3oec.resample(process_3oec.make_timeseries(self.data_path_test), self.sampling_freq)
    
        self.fl1 = process_3oec.get_first_piece(self.df_fl)
        self.fl1_mean = self.fl1.mean().values
        self.fl1_std = self.fl1.std().values
        self.fl2 = process_3oec.get_second_piece(self.df_fl)
        self.fl2_mean = self.fl2.mean().values
        self.fl2_std = self.fl2.std().values
        self.fl3 = process_3oec.get_third_piece(self.df_fl)
        self.fl3_mean = self.fl3.mean().values
        self.fl3_std = self.fl3.std().values
        self.fl4 = process_3oec.get_fourth_piece(self.df_fl)
        self.fl4_mean = self.fl4.mean().values
        self.fl4_std = self.fl4.std().values


        # create train sequences (FL1-4, SB1-2)
        self.train_seq1, self.train_labels1 = create_sequences_single_step(standardize_piece(self.fl1).values, self.sequence_len)
        self.train_seq2, self.train_labels2 = create_sequences_single_step(standardize_piece(self.fl2).values, self.sequence_len)
        self.train_seq3, self.train_labels3 = create_sequences_single_step(standardize_piece(self.fl3).values, self.sequence_len)
        self.train_seq4, self.train_labels4 = create_sequences_single_step(standardize_piece(self.fl4).values, self.sequence_len)
        self.train_seq5, self.train_labels5 = create_sequences_single_step(standardize_piece(self.sb1).values, self.sequence_len)
        self.train_seq6, self.train_labels6 = create_sequences_single_step(standardize_piece(self.sb2).values, self.sequence_len)

        # create TEST (SB3-4) sequences
        self.sb_seq3, self.sb_labels3 = create_sequences_single_step(standardize_piece(self.sb3).values, self.sequence_len)
        self.sb_seq4, self.sb_labels4 = create_sequences_single_step(standardize_piece(self.sb4).values, self.sequence_len)
        
        # create train tensors
        self.seq1_tensor = make_sequence_tensor(self.train_seq1, self.device)
        self.labels1_tensor = make_sequence_tensor(self.train_labels1, self.device)
        self.seq2_tensor = make_sequence_tensor(self.train_seq2, self.device)
        self.labels2_tensor = make_sequence_tensor(self.train_labels2, self.device)
        self.seq3_tensor = make_sequence_tensor(self.train_seq3, self.device)
        self.labels3_tensor = make_sequence_tensor(self.train_labels3, self.device)
        self.seq4_tensor = make_sequence_tensor(self.train_seq4, self.device)
        self.labels4_tensor = make_sequence_tensor(self.train_labels4, self.device)
        self.seq5_tensor = make_sequence_tensor(self.train_seq5, self.device)
        self.labels5_tensor = make_sequence_tensor(self.train_labels5, self.device)
        self.seq6_tensor = make_sequence_tensor(self.train_seq6, self.device)
        self.labels6_tensor = make_sequence_tensor(self.train_labels6, self.device)

        # create test tensors
        self.sb3_tensor = make_sequence_tensor(self.sb_seq3, self.device)
        self.sb_labels3_tensor = make_sequence_tensor(self.sb_labels3, self.device)
        self.sb4_tensor = make_sequence_tensor(self.sb_seq4, self.device)
        self.sb_labels4_tensor = make_sequence_tensor(self.sb_labels4, self.device)

        
    def set_train_data(self):
        self.train_seqs = [
            self.seq1_tensor,
            self.seq2_tensor,
            self.seq3_tensor,
            self.seq4_tensor,
            self.seq5_tensor,
            self.seq6_tensor,
        ]
        self.train_lbls = [
            self.labels1_tensor,
            self.labels2_tensor,
            self.labels3_tensor,
            self.labels4_tensor,
            self.labels5_tensor,
            self.labels6_tensor,
        ]
    
    def train_model(self, num_epochs: int,
                     model_label: str):
        """
        Trains the model without using batches, i.e. one sequence per iter

        Args:
            num_epochs (int)
            model_label (str)
        """
        self.set_train_data()

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        for epoch in range(num_epochs):
            for seq, lbls in zip(self.train_seqs, self.train_lbls):
                for k in range(len(seq)):
                    model.train()
                    optimizer.zero_grad()
                    pred = model(seq[k])

                    loss = criterion(pred, lbls[k])
                    loss.backward()
                    optimizer.step()
            
            train_losses.append(loss.item())
            if epoch % 10 == 0:
                print(f'Model {model_label}, Epoch {epoch}, Train Loss {loss.item()}')
        
        return model, train_losses

    def train_model_es(self, num_epochs: int, model_label: str, patience: int = 10):
        """
        Trains the model with early stopping based on training loss only.

        Args:
        num_epochs (int): Max number of epochs
        model_label (str): Label for the model
        patience (int): Epochs to wait for improvement before stopping
        """
        self.set_train_data()

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        best_train_loss = float('inf')
        epochs_no_improve = 0
        best_model_state = None

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            total_steps = 0
            for seq, lbls in zip(self.train_seqs, self.train_lbls):
                for k in range(len(seq)):
                    optimizer.zero_grad()
                    pred = model(seq[k])
                    loss = criterion(pred, lbls[k])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    total_steps += 1
        
            avg_train_loss = epoch_loss / total_steps if total_steps else 0.0
            train_losses.append(avg_train_loss)

            if epoch % 10 == 0:
                print(f'Model {model_label}, Epoch {epoch}, Avg Train Loss {avg_train_loss:.4f}')
        
            # ---- Early Stopping ----
            if avg_train_loss < best_train_loss - 1e-6:  # small delta to avoid stopping on tiny fluctuations
                best_train_loss = avg_train_loss
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}. Best Train Loss: {best_train_loss:.4f}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        return model, train_losses
    
    def rolling_average_model(self, test_seq, window_size):
        """
        Simple moving avg model that serves as a baseline
        """
        
        preds = []
        test_seq = test_seq.cpu().numpy()
        for i in range(len(test_seq)):
            pred = np.mean(test_seq[i:i+window_size])
            preds.append(pred)
        return preds
    
    def test_model(self, model: nn.Module, model_lbl: str, seed: int, test_seq: torch.FloatTensor, test_lbl: torch.FloatTensor,
                   prefix: str, ds_lbl: str, train_losses: list,  save_dir: str):
        """
        Test a trained model

        Args:
            model (nn.Module)
            model_lbl (str)
            test_seq (torch.FloatTensor)
            test_lbl (torch.FloatTensor)
            ds_lbl (str)
        """
        preds = []
        model.eval()
        
        with torch.no_grad():
            for i in range(len(test_seq)):
                pred = model(test_seq[i])
                
                if self.device != 'cpu':
                    preds.append(pred.cpu())
                else:
                    preds.append(pred)
        

        baseline_preds = self.rolling_average_model(test_seq, 12)
        plotting.plot_preds.plot_preds_from_device(preds, baseline_preds, test_lbl, filename_prefix=prefix, top_dir=save_dir)


        # criterion2(torch.FloatTensor(y_pred).to(device=device), test_tensor_labels_2.squeeze(1)).item()

        l1_smooth_error = nn.SmoothL1Loss()
        l1_smooth_value = l1_smooth_error(torch.FloatTensor(preds).to(device='cpu'), test_lbl.squeeze(1).to(device='cpu')).item()
        l1_smooth_value_baseline = l1_smooth_error(torch.FloatTensor(baseline_preds).to(device='cpu'), test_lbl.squeeze(1).to(device='cpu')).item()

        unnormalized_preds, unnormalized_lbls = self.unnormalize(preds, ds_lbl)
        unnormalized_preds_baseline, _ = self.unnormalize(baseline_preds, ds_lbl)
        error = MARE(unnormalized_preds, unnormalized_lbls)
        error_baseline = MARE(unnormalized_preds_baseline, unnormalized_lbls)
        csv_filepath = write_csv(model_lbl, ds_lbl, seed, error.item(), error_baseline.item(), l1_smooth_error,
                                 l1_smooth_value, l1_smooth_value_baseline, train_losses, save_dir)
        return csv_filepath


    
    def run_experiment(self, num_runs: int, save_dir : str ='./out/exp1', early_stop: bool = False):
        os.mkdir(save_dir)
        l1_smooth_error = nn.SmoothL1Loss()
        model_label = 'model_all_train'
        self.process_data()
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            if early_stop:
                cur_model, train_losses = self.train_model_es(self.num_train_epochs, model_label)
            else:
                cur_model, train_losses = self.train_model(self.num_train_epochs, model_label)
            self.set_test_data()
            for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                prefix = f'{model_label},seed_{i},{ds_lbl}'
                csv_filepath = self.test_model(cur_model, model_label, i, test_seq, test_lbl, prefix, ds_lbl, train_losses, save_dir)
        
        write_stats_mare(csv_filepath, save_dir)
        write_stats_crit(csv_filepath, save_dir, l1_smooth_error)
        plot_MARE(csv_filepath, save_dir)
        plot_smooth_l1_loss(csv_filepath, save_dir)

                
    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        preds = torch.FloatTensor(preds)
        preds = preds.cpu()
        if ds_lbl == 'sb_dataset3':
            y_hat = preds.numpy() * self.sb3_std + self.sb3_mean
            unnormed_labels = self.sb_labels3_tensor.squeeze(1).cpu().numpy() * self.sb3_std + self.sb3_mean
        elif ds_lbl == 'sb_dataset4':
            y_hat = preds.numpy() * self.sb4_std + self.sb4_mean
            unnormed_labels = self.sb_labels4_tensor.squeeze(1).cpu().numpy() * self.sb4_std + self.sb4_mean
        else:
            print('unnormalize_pred(): Incorrect dataset label')
            return False
        return y_hat, unnormed_labels


    def set_test_data(self, test_only: bool=False):
        # if not test_only:
        #     self.dataset_lbls = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
        #     self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor, self.seq4_tensor]
        #     self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor, self.labels4_tensor]
        #     return True
        self.dataset_lbls = ['sb_dataset3', 'sb_dataset4']
        self.test_seqs = [self.sb3_tensor, self.sb4_tensor]
        self.test_lbls = [self.sb_labels3_tensor, self.sb_labels4_tensor]
        # if model_lbl == 'model_1':
        #     self.dataset_lbls = ['dataset2', 'dataset3', 'dataset4']
        #     self.test_seqs = [self.seq2_tensor, self.seq3_tensor, self.seq4_tensor]
        #     self.test_lbls = [self.labels2_tensor, self.labels3_tensor, self.labels4_tensor]
        # elif model_lbl == 'model_2':
        #     self.dataset_lbls = ['dataset1', 'dataset3', 'dataset4']
        #     self.test_seqs = [self.seq1_tensor, self.seq3_tensor, self.seq4_tensor]
        #     self.test_lbls = [self.labels1_tensor, self.labels3_tensor, self.labels4_tensor]
        # elif model_lbl == 'model_3':
        #     self.dataset_lbls = ['dataset1', 'dataset2', 'dataset4']
        #     self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq4_tensor]
        #     self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels4_tensor]
        # elif model_lbl == 'model_4':
        #     self.dataset_lbls = ['dataset1', 'dataset2', 'dataset3']
        #     self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor]
        #     self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor]
        # else:
        #     print('Wrong model label')
        #     return False
