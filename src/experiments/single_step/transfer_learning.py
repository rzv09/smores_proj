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

class TransferLearning():
    def __init__(self, data_path: str, num_train_epochs: int, sampling_freq: str, 
                 sequence_len: int, device: str):
        self.data_path = data_path
        self.num_train_epochs = num_train_epochs
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device
    
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


        # create sequences
        self.seq1, self.labels1 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.first_piece).values,
                                                                       self.sequence_len)
        self.seq2, self.labels2 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.second_piece).values,
                                                                      self.sequence_len)
        self.seq3, self.labels3 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.third_piece).values,
                                                                      self.sequence_len)
        self.seq4, self.labels4 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.fourth_piece).values,
                                                                      self.sequence_len)
        
        # create train tensors
        self.seq1_tensor = sequences_3oec.make_sequence_tensor(self.seq1, self.device)
        self.labels1_tensor = sequences_3oec.make_sequence_tensor(self.labels1, self.device)
        self.seq2_tensor = sequences_3oec.make_sequence_tensor(self.seq2, self.device)
        self.labels2_tensor = sequences_3oec.make_sequence_tensor(self.labels2, self.device)
        self.seq3_tensor = sequences_3oec.make_sequence_tensor(self.seq3, self.device)
        self.labels3_tensor = sequences_3oec.make_sequence_tensor(self.labels3, self.device)
        self.seq4_tensor = sequences_3oec.make_sequence_tensor(self.seq4, self.device)
        self.labels4_tensor = sequences_3oec.make_sequence_tensor(self.labels4, self.device)
    
    def set_train_data(self, model_label: str):
        if model_label == 'model_123':
            self.train_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor]
            self.train_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor]
        elif model_label == 'model_12':
            self.train_seqs = [self.seq1_tensor, self.seq2_tensor]
            self.train_lbls = [self.labels1_tensor, self.labels2_tensor]
        elif model_label == 'model_124':
            self.train_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq4_tensor]
            self.train_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels4_tensor]
        else:
            print(f"set_train_data(): Model {model_label} does not exist")
            return False
        
    def train_model(self, num_epochs: int, model_label: str):
        """
        Train the model sequentially using transfer learning

        Args:
            num_epochs (int)
            model_label (str)
        """
        self.set_train_data(model_label)

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        for seq, lbl in zip(self.train_seqs, self.train_lbls):
            for epoch in range(num_epochs):
                for k in range(len(seq)):
                    model.train()
                    optimizer.zero_grad()
                    pred = model(seq[k])

                    loss = criterion(pred, lbl[k])
                    loss.backward()
                    optimizer.step()
                
                train_losses.append(loss.item())
                if epoch % 10 == 0:
                    print(f"Model {model_label}, Epoch {epoch}, Train Loss {loss.item()}")
        return model, train_losses
    
    def set_test_data(self, model_lbl: str, test_only: bool = False):
        if not test_only:
            self.dataset_lbls = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
            self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor, self.seq4_tensor]
            self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor, self.labels4_tensor]   
            return True         
    
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
        model_labels = ['model_123', 'model_12', 'model_124']
        self.process_data()
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            for model_lbl in model_labels:
                cur_model, train_losses = self.train_model(self.num_train_epochs, model_lbl)
                self.set_test_data(model_lbl)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f'{model_lbl},seed_{i},{ds_lbl}'
                    csv_filepath = self.test_model(cur_model, model_lbl, i, test_seq, test_lbl, prefix, ds_lbl, train_losses, save_dir)
        write_stats(csv_filepath, save_dir)

