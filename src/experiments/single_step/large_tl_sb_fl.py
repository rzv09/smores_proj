import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import itertools

import data_processing.santa_barbara.process_sb as process_sb
from data_processing.data_utils import standardize_piece, create_sequences_single_step, make_sequence_tensor 
import data_processing.three_oec.process_3oec as process_3oec
import data_processing.three_oec.sequences_3oec as sequences_3oec 
import models.lstm.single_step as lstm_single_step
import plotting.plot_preds
import utils
import utils.set_seeds
from metrics.np.regression import MARE
from utils.write_metrics import write_csv, write_stats_mare, write_stats_crit
from plotting.plot_smoothl1loss import plot_smooth_l1_loss
from plotting.plot_mare import plot_MARE

class TransferLearningSBFL():
    def __init__(self, data_path_sb, data_path_fl, num_train_epochs, sampling_freq, sequence_len, device):
        self.data_path_sb = data_path_sb
        self.data_path_fl = data_path_fl
        self.num_train_epochs = num_train_epochs
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device

    def process_data(self):
        # preprocess data and resample to sampling freq
        self.df_sb = process_sb.make_timeseries(self.data_path_sb)
        self.df_fl = process_3oec.resample(process_3oec.make_timeseries(self.data_path_fl), self.sampling_freq)

        pieces = {
            "sb1": process_sb.get_first_piece(self.df_sb),
            "sb2": process_sb.get_second_piece(self.df_sb),
            "sb3": process_sb.get_third_piece(self.df_sb),
            "sb4": process_sb.get_fourth_piece(self.df_sb),
            "fl1": process_3oec.get_first_piece(self.df_fl),
            "fl2": process_3oec.get_second_piece(self.df_fl),
            "fl3": process_3oec.get_third_piece(self.df_fl),
            "fl4": process_3oec.get_fourth_piece(self.df_fl),
        }

        self.dataset_info = {}
        for label, piece in pieces.items():
            mean = piece.mean().values
            std = piece.std().values
            seqs, labels = create_sequences_single_step(standardize_piece(piece).values, self.sequence_len)
            seq_tensor = make_sequence_tensor(seqs, self.device)
            lbl_tensor = make_sequence_tensor(labels, self.device)
            self.dataset_info[label] = {
                "mean": mean,
                "std": std,
                "seq_tensor": seq_tensor,
                "lbl_tensor": lbl_tensor,
            }
        
    def set_train_data(self, train_labels):
        self.train_seqs = [self.dataset_info[label]["seq_tensor"] for label in train_labels]
        self.train_lbls = [self.dataset_info[label]["lbl_tensor"] for label in train_labels]
    
    def train_model(self, num_epochs: int, model_label: str, train_labels):
        """
        Trains the model without using batches, i.e. one sequence per iter

        Args:
            num_epochs (int)
            model_label (str)
        """
        self.set_train_data(train_labels)

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        for epoch in range(num_epochs):
            for k in range(len(self.train_seqs)):
                model.train()
                optimizer.zero_grad()
                pred = model(self.train_seqs[k])

                loss = criterion(pred, self.train_lbls[k])
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
            if epoch % 10 == 0:
                print(f'Model {model_label}, Epoch {epoch}, Train Loss {loss.item()}')
        
        return model, train_losses

    def train_model_es(self, num_epochs: int, model_label: str, train_labels, patience: int = 10):
        """
        Trains the model with early stopping based on training loss only.

        Args:
        num_epochs (int): Max number of epochs
        model_label (str): Label for the model
        patience (int): Epochs to wait for improvement before stopping
        """
        self.set_train_data(train_labels)

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        # train_losses = []

        # best_train_loss = float('inf')
        # epochs_no_improve = 0
        # best_model_state = None
        full_train_losses = []
        for seq, lbls in zip(self.train_seqs, self.train_lbls):
            train_losses = []
            best_train_loss = float('inf')
            epochs_no_improve = 0
            best_model_state = None
            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0.0
                for k in range(len(seq)):
                    optimizer.zero_grad()
                    pred = model(seq[k])
                    loss = criterion(pred, lbls[k])
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                # incorrect because adds last loss
                # full_train_losses.append(loss.item())
        
                avg_train_loss = epoch_loss / len(seq)
                train_losses.append(avg_train_loss)
                full_train_losses.append(avg_train_loss)

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

        return model, full_train_losses
    
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
    

    def run_experiment(self, num_runs: int, save_dir : str ='./out/exp1', early_stop: bool = False):
        os.mkdir(save_dir)
        l1_smooth_error = nn.SmoothL1Loss()
        self.process_data()
        dataset_labels = ["sb1", "sb2", "sb3", "sb4", "fl1", "fl2", "fl3", "fl4"]
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            for perm in itertools.permutations(dataset_labels, 8):
                train_labels = perm[:4]
                test_labels = perm[4:]
                model_lbl = f"model_{'-'.join(train_labels)}_to_{'-'.join(test_labels)}"
                if early_stop:
                    cur_model, train_losses = self.train_model_es(self.num_train_epochs, model_lbl, train_labels)
                else:
                    cur_model, train_losses = self.train_model(self.num_train_epochs, model_lbl, train_labels)
                self.set_test_data(test_labels)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f'{model_lbl},seed_{i},{ds_lbl}'
                    csv_filepath = self.test_model(cur_model, model_lbl, i, test_seq, test_lbl, prefix, ds_lbl, train_losses, save_dir)
        
        write_stats_mare(csv_filepath, save_dir)
        write_stats_crit(csv_filepath, save_dir, l1_smooth_error)
        plot_smooth_l1_loss(csv_filepath, save_dir)
        plot_MARE(csv_filepath, save_dir)
                
    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        preds = torch.FloatTensor(preds)
        preds = preds.cpu()
        info = self.dataset_info.get(ds_lbl)
        if info is None:
            print('unnormalize_pred(): Incorrect dataset label')
            return False
        y_hat = preds.numpy() * info["std"] + info["mean"]
        unnormed_labels = info["lbl_tensor"].squeeze(1).cpu().numpy() * info["std"] + info["mean"]
        return y_hat, unnormed_labels


    def set_test_data(self, test_labels, test_only: bool=False):
        self.dataset_lbls = list(test_labels)
        self.test_seqs = [self.dataset_info[label]["seq_tensor"] for label in test_labels]
        self.test_lbls = [self.dataset_info[label]["lbl_tensor"] for label in test_labels]
