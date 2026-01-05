import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import os

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

class FederatedLearningSBFL():
    def __init__(self, data_path_sb, data_path_fl, num_train_epochs, num_rounds,
                 agg_method, sampling_freq, sequence_len, device):
        self.data_path_sb = data_path_sb
        self.data_path_fl = data_path_fl
        self.num_train_epochs = num_train_epochs
        self.num_rounds = num_rounds
        self.agg_method = agg_method
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device
        
        self.num_clients = 4

    def process_data(self):

        # preprocess data and resample to sampling freq
        self.df_sb = process_sb.make_timeseries(self.data_path_sb)
        self.df_fl = process_3oec.resample(process_3oec.make_timeseries(self.data_path_fl), self.sampling_freq)

        # For training, use SB1 SB2, FL3, FL4 split
        self.sb1 = process_sb.get_first_piece(self.df_sb)
        self.sb1_mean = self.sb1.mean().values
        self.sb1_std = self.sb1.std().values
        self.sb2 = process_sb.get_second_piece(self.df_sb)
        self.sb2_mean = self.sb2.mean().values
        self.sb2_std = self.sb2.std().values
        self.fl3 = process_3oec.get_third_piece(self.df_fl)
        self.fl3_mean = self.fl3.mean().values
        self.fl3_std = self.fl3.std().values
        self.fl4 = process_3oec.get_fourth_piece(self.df_fl)
        self.fl4_mean = self.fl4.mean().values
        self.fl4_std = self.fl4.std().values

        # For testing, use SB3, FL1, SB4, FL2 sequences (in order)
        self.sb3 = process_sb.get_third_piece(self.df_sb)
        self.sb3_mean = self.sb3.mean().values
        self.sb3_std = self.sb3.std().values
        self.fl1 = process_3oec.get_first_piece(self.df_fl)
        self.fl1_mean = self.fl1.mean().values
        self.fl1_std = self.fl1.std().values
        self.sb4 = process_sb.get_fourth_piece(self.df_sb)
        self.sb4_mean = self.sb4.mean().values
        self.sb4_std = self.sb4.std().values
        self.fl2 = process_3oec.get_second_piece(self.df_fl)
        self.fl2_mean = self.fl2.mean().values
        self.fl2_std = self.fl2.std().values

        # create train sequences
        self.train_seq1, self.train_labels1 = create_sequences_single_step(standardize_piece(self.sb1).values, self.sequence_len)
        self.train_seq2, self.train_labels2 = create_sequences_single_step(standardize_piece(self.sb2).values, self.sequence_len)
        self.train_seq3, self.train_labels3 = create_sequences_single_step(standardize_piece(self.fl3).values, self.sequence_len)
        self.train_seq4, self.train_labels4 = create_sequences_single_step(standardize_piece(self.fl4).values, self.sequence_len)
        
        # create train tensors
        self.seq1_tensor = make_sequence_tensor(self.train_seq1, self.device)
        self.labels1_tensor = make_sequence_tensor(self.train_labels1, self.device)
        self.seq2_tensor = make_sequence_tensor(self.train_seq2, self.device)
        self.labels2_tensor = make_sequence_tensor(self.train_labels2, self.device)
        self.seq3_tensor = make_sequence_tensor(self.train_seq3, self.device)
        self.labels3_tensor = make_sequence_tensor(self.train_labels3, self.device)
        self.seq4_tensor = make_sequence_tensor(self.train_seq4, self.device)
        self.labels4_tensor = make_sequence_tensor(self.train_labels4, self.device)

        # create test sequences and tensors
        self.test_seq1, self.test_labels1 = create_sequences_single_step(standardize_piece(self.sb3).values, self.sequence_len)
        self.test_seq2, self.test_labels2 = create_sequences_single_step(standardize_piece(self.fl1).values, self.sequence_len)
        self.test_seq3, self.test_labels3 = create_sequences_single_step(standardize_piece(self.sb4).values, self.sequence_len)
        self.test_seq4, self.test_labels4 = create_sequences_single_step(standardize_piece(self.fl2).values, self.sequence_len)

        self.test_seq1_tensor = make_sequence_tensor(self.test_seq1, self.device)
        self.test_labels1_tensor = make_sequence_tensor(self.test_labels1, self.device)
        self.test_seq2_tensor = make_sequence_tensor(self.test_seq2, self.device)
        self.test_labels2_tensor = make_sequence_tensor(self.test_labels2, self.device)
        self.test_seq3_tensor = make_sequence_tensor(self.test_seq3, self.device)
        self.test_labels3_tensor = make_sequence_tensor(self.test_labels3, self.device)
        self.test_seq4_tensor = make_sequence_tensor(self.test_seq4, self.device)
        self.test_labels4_tensor = make_sequence_tensor(self.test_labels4, self.device)
    
    def set_train_data(self, model_label: str):
        if model_label == 'model_1':
            self.train_seq = self.seq1_tensor
            self.train_lbls = self.labels1_tensor
        elif model_label == 'model_2':
            self.train_seq = self.seq2_tensor
            self.train_lbls = self.labels2_tensor
        elif model_label == 'model_3':
            self.train_seq = self.seq3_tensor
            self.train_lbls = self.labels3_tensor
        elif model_label == 'model_4':
            self.train_seq = self.seq4_tensor
            self.train_lbls = self.labels4_tensor
        else:
            return False

        return 1

    def fed_avg(self, models: list):
        """
        Aggregate the models using averaging with
        equal weights

        Args:
            models (list) : list of models to be aggregated
        """
        global_model = models[0]
        avg_state_dict = global_model.state_dict()

        with torch.no_grad():
            for key in avg_state_dict.keys():
                avg_state_dict[key] = torch.stack([m.state_dict()[key] for m in models], dim=0).mean(dim=0)
        global_model.load_state_dict(avg_state_dict)
        return global_model

    def fed_avg_weighted(self, models: list, client_ds_sizes: list):
        """
        Aggregate the models using Federated Avg

        Args:
            models (list) : list of models to be aggregated
            client_ds_sizes (list) : list of client dataset sizes
        """
        global_model = models[0]
        avg_state_dict = global_model.state_dict()

        total_ds_size = sum(client_ds_sizes)
        weights = [size / total_ds_size for size in client_ds_sizes]

        with torch.no_grad():
            for key in avg_state_dict.keys():
                avg_state_dict[key] = torch.stack([models[i].state_dict()[key] * weights[i] for i in range(len(models))], dim=0).sum(dim=0)

        global_model.load_state_dict(avg_state_dict)
        return global_model

    def train_model(self, model_label: str):
        """
        Trains the model without using batches, i.e. one sequence per iter

        Args:
            num_epochs (int)
            model_label (str)
        """
        self.set_train_data(model_label)

        global_model = lstm_single_step.create_lstm_single_step()
        global_model.to(self.device)
        criterion = nn.SmoothL1Loss()

        train_losses = {}

        for round in range(self.num_rounds):
            print(f"\nFL Round {round+1}")
            local_models = []

            for i in range(self.num_clients):
                client_losses = []
                # print(f"Client {i+1}")
                local_model = lstm_single_step.create_lstm_single_step()
                local_model.to(self.device)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.Adam(local_model.parameters(), lr=1e-4)
                local_model.train()

                model_lbl = f"model_{i+1}"
                self.set_train_data(model_lbl)

                for _ in range(self.num_train_epochs):
                    
                    for j in range(len(self.train_seq)):
                        optimizer.zero_grad()
                        pred = local_model(self.train_seq[j])
                        loss = criterion(pred, self.train_lbls[j])
                        loss.backward()
                        optimizer.step()
                print(f"Round {round+1}, Client {i+1}, Train Loss {loss.item()}")
                local_models.append(local_model)
                client_losses.append(loss.item())
            train_losses[model_label] = client_losses

            # aggregate at the end of each round
            if (self.agg_method == 'fedavg'):
                global_model = self.fed_avg_weighted(local_models,
                                                 [len(self.labels1_tensor), len(self.labels2_tensor), len(self.labels3_tensor)])
            elif (self.agg_method == 'avg'):
                global_model = self.fed_avg(local_models)
            else:
                print(f"train_model() : unrecognized aggregation method")
                return 
        return global_model, train_losses
    

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
    
    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        preds = torch.FloatTensor(preds)
        preds = preds.cpu()
        if ds_lbl == 'sb_dataset3':
            y_hat = preds.numpy() * self.sb3_std + self.sb3_mean
            unnormed_labels = self.test_labels1_tensor.squeeze(1).cpu().numpy() * self.sb3_std + self.sb3_mean
        elif ds_lbl == 'fl_dataset1':
            y_hat = preds.numpy() * self.fl1_std + self.fl1_mean
            unnormed_labels = self.test_labels2_tensor.squeeze(1).cpu().numpy() * self.fl1_std + self.fl1_mean
        elif ds_lbl == 'sb_dataset4':
            y_hat = preds.numpy() * self.sb4_std + self.sb4_mean
            unnormed_labels = self.test_labels3_tensor.squeeze(1).cpu().numpy() * self.sb4_std + self.sb4_mean
        elif ds_lbl == 'fl_dataset2':
            y_hat = preds.numpy() * self.fl2_std + self.fl2_mean
            unnormed_labels = self.test_labels4_tensor.squeeze(1).cpu().numpy() * self.fl2_std + self.fl2_mean
        else:
            print('unnormalize_pred(): Incorrect dataset label')
            return False
        return y_hat, unnormed_labels


    def set_test_data(self, model_lbl: str, test_only: bool=False):
        self.dataset_lbls = ['sb_dataset3', 'fl_dataset1', 'sb_dataset4', 'fl_dataset2']
        self.test_seqs = [self.test_seq1_tensor, self.test_seq2_tensor, self.test_seq3_tensor, self.test_seq4_tensor]
        self.test_lbls = [self.test_labels1_tensor, self.test_labels2_tensor, self.test_labels3_tensor, self.test_labels4_tensor]


    def run_experiment(self, num_runs: int, save_dir : str ='./out/exp1'):
        os.mkdir(save_dir)
        l1_smooth_error = nn.SmoothL1Loss()
        model_labels = ['model_FL_sb1sb2fl3fl4']
        self.process_data()
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            for model_lbl in model_labels:
                cur_model, train_losses = self.train_model(model_lbl)
                self.set_test_data(model_lbl)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f'{model_lbl},seed_{i},{ds_lbl}'
                    csv_filepath = self.test_model(cur_model, model_lbl, i, test_seq, test_lbl, prefix, ds_lbl, train_losses, save_dir)
        write_stats_mare(csv_filepath, save_dir)
        write_stats_crit(csv_filepath, save_dir, l1_smooth_error)
        plot_smooth_l1_loss(csv_filepath, save_dir)
        plot_MARE(csv_filepath, save_dir)