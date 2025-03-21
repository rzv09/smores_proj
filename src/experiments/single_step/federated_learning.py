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
import models.lstm.single_step as lstm_single_step
import plotting.plot_preds
import utils
import utils.set_seeds
from metrics.np.regression import MARE
from utils.write_metrics import write_csv, write_stats

class FederatedLearning():
    def __init__(self, data_path, num_train_epochs, num_rounds,
                 agg_method, sampling_freq, sequence_len, device):
        self.data_path = data_path
        self.num_train_epochs = num_train_epochs
        self.num_rounds = num_rounds
        self.agg_method = agg_method
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device

        self.num_clients = 3

    def process_data(self):
        # preprocess data and resample to sampling freq
        self.df = process_3oec.resample(process_3oec.make_timeseries(self.data_path), self.sampling_freq)

        self.first_piece = process_3oec.get_first_piece(self.df)
        self.first_piece_mean = self.first_piece.mean().values
        self.first_piece_std = self.first_piece.std().values
        self.second_piece = process_3oec.get_second_piece(self.df)
        self.second_piece_mean = self.second_piece.mean().values
        self.second_piece_std = self.second_piece.std().values
        self.third_piece = process_3oec.get_third_piece(self.df)
        self.third_piece_mean = self.third_piece.mean().values
        self.third_piece_std = self.third_piece.std().values
        self.fourth_piece = process_3oec.get_fourth_piece(self.df)
        self.fourth_piece_mean = self.fourth_piece.mean().values
        self.fourth_piece_std = self.fourth_piece.std().values


        # create train sequences
        self.train_seq1, self.train_labels1 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.first_piece).values,
                                                                       self.sequence_len)
        self.train_seq2, self.train_labels2 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.second_piece).values,
                                                                      self.sequence_len)
        self.train_seq3, self.train_labels3 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.third_piece).values,
                                                                      self.sequence_len)
        self.train_seq4, self.train_labels4 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.fourth_piece).values,
                                                                      self.sequence_len)
        
        # create train tensors
        self.seq1_tensor = sequences_3oec.make_sequence_tensor(self.train_seq1, self.device)
        self.labels1_tensor = sequences_3oec.make_sequence_tensor(self.train_labels1, self.device)
        self.seq2_tensor = sequences_3oec.make_sequence_tensor(self.train_seq2, self.device)
        self.labels2_tensor = sequences_3oec.make_sequence_tensor(self.train_labels2, self.device)
        self.seq3_tensor = sequences_3oec.make_sequence_tensor(self.train_seq3, self.device)
        self.labels3_tensor = sequences_3oec.make_sequence_tensor(self.train_labels3, self.device)
        self.seq4_tensor = sequences_3oec.make_sequence_tensor(self.train_seq4, self.device)
        self.labels4_tensor = sequences_3oec.make_sequence_tensor(self.train_labels4, self.device)
    
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
        optimizer = torch.optim.Adam(global_model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = {}

        for round in range(self.num_rounds):
            print(f"\nFL Round {round+1}")
            local_models = []

            for i in range(self.num_clients):
                client_losses = []
                print(f"Client {i+1}")
                local_model = lstm_single_step.create_lstm_single_step()
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.Adam(local_model.parameters(), lr=1e-4)
                local_model.train()

                model_lbl = f"model_{i+1}"

                for _ in range(self.num_train_epochs):
                    self.set_train_data(model_lbl)
                    for j in range(len(self.train_seq)):
                        optimizer.zero_grad()
                        pred = local_model(self.train_seq[j])
                        loss = criterion(pred, self.train_lbls[k])
                        loss.backward()
                        optimizer.step()
                print(f"Round {round+1}, Client {i+1}, Train Loss {loss.item()}")
                local_models.append(local_model)
                client_losses.append(loss.item())
            train_losses[model_label] = client_losses
        
        if (self.agg_method == 'fedavg'):
            global_model = self.fed_avg_weighted(local_models,
                                                 [len(self.labels1_tensor), len(self.labels2_tensor), len(self.labels3_tensor)])
        elif (self.agg_method == 'avg'):
            global_model = self.fed_avg(local_models)
        else:
            print(f"train_model() : unrecognized aggregation method")
            return 
        return global_model, client_losses
    
    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        preds = torch.FloatTensor(preds)
        preds = preds.cpu()
        if ds_lbl == 'dataset1':
            y_hat = preds.numpy() * self.first_piece_std + self.first_piece_mean
            unnormed_labels = self.labels1_tensor.squeeze(1).cpu().numpy() * self.first_piece_std + self.first_piece_mean
        elif ds_lbl == 'dataset2':
            y_hat = preds.numpy() * self.second_piece_std + self.second_piece_mean
            unnormed_labels = self.labels2_tensor.squeeze(1).cpu().numpy() * self.second_piece_std + self.second_piece_mean
        elif ds_lbl == 'dataset3':
            y_hat = preds.numpy() * self.third_piece_std + self.third_piece_mean
            unnormed_labels = self.labels3_tensor.squeeze(1).cpu().numpy() * self.third_piece_std + self.third_piece_mean
        elif ds_lbl == 'dataset4':
            y_hat = preds.numpy() * self.fourth_piece_std + self.fourth_piece_mean
            unnormed_labels = self.labels4_tensor.squeeze(1).cpu().numpy() * self.fourth_piece_std + self.fourth_piece_mean
        else:
            print('unnormalize_pred(): Incorrect dataset label')
            return False
        return y_hat, unnormed_labels


    def set_test_data(self, model_lbl: str, test_only: bool=False):
        if not test_only:
            self.dataset_lbls = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
            self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor, self.seq4_tensor]
            self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor, self.labels4_tensor]
            return True
    
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
        

        plotting.plot_preds.plot_preds_from_device(preds, test_lbl, filename_prefix=prefix, top_dir=save_dir)

        unnormalized_preds, unnormalized_lbls = self.unnormalize(preds, ds_lbl)
        error = MARE(unnormalized_preds, unnormalized_lbls)
        
        csv_filepath = write_csv(model_lbl, ds_lbl, seed, error.item(), train_losses, save_dir)
        return csv_filepath
    
    def run_experiment(self, num_runs: int, save_dir : str ='./out/exp1'):
        os.mkdir(save_dir)
        model_labels = ['model_123']
        self.process_data()
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            for model_lbl in model_labels:
                cur_model, train_losses = self.train_model(model_lbl)
                self.set_test_data(model_lbl)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f'{model_lbl},seed_{i},{ds_lbl}'
                    csv_filepath = self.test_model(cur_model, model_lbl, i, test_seq, test_lbl, prefix, ds_lbl, train_losses, save_dir)
        write_stats(csv_filepath, save_dir)