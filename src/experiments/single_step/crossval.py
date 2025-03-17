import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

import data_processing.three_oec.process_3oec as process_3oec
import data_processing.three_oec.sequences_3oec as sequences_3oec 
import models.lstm.single_step as lstm_single_step
import plotting.plot_preds
import utils
import utils.set_seeds
from metrics.np.regression import MARE

class CrossVal():
    def __init__(self, data_path, num_train_epochs, sampling_freq, sequence_len, device):
        self.data_path = data_path
        self.num_train_epochs = num_train_epochs
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device

    def process_data(self):
        # preprocess data and resample to sampling freq
        self.df = process_3oec.resample(process_3oec.make_timeseries(self.data_path), self.sampling_freq)

        self.first_piece = process_3oec.get_first_piece(self.df)
        self.first_piece_mean = self.first_piece.mean()
        self.first_piece_std = self.first_piece.std()
        self.second_piece = process_3oec.get_second_piece(self.df)
        self.second_piece_mean = self.second_piece.mean()
        self.second_piece_std = self.second_piece.std()
        self.third_piece = process_3oec.get_third_piece(self.df)
        self.third_piece_mean = self.third_piece.mean()
        self.third_piece_std = self.third_piece.std()
        self.fourth_piece = process_3oec.get_fourth_piece(self.df)
        self.fourth_piece_mean = self.fourth_piece.mean()
        self.fourth_piece_std = self.fourth_piece_std()


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
    
    def train_model(self, num_epochs: int,
                     model_label: str):
        """
        Trains the model without using batches, i.e. one sequence per iter

        Args:
            num_epochs (int)
            model_label (str)
            train_seq (torch.FloatTensor)
            train_lbl (torch.FloatTensor)
        """
        self.set_train_data(model_label)

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        for epoch in range(num_epochs):
            for k in range(len(self.train_seq)):
                model.train()
                optimizer.zero_grad()
                pred = model(self.train_seq[k])

                loss = criterion(pred, self.train_lbls[k])
                loss.backward()
                optimizer.step()
        # if epoch % 10 == 0:
            print(f'Model {model_label}, Epoch {epoch}, Train Loss {loss.item()}')
        
        return model
    
    def test_model(self, model: nn.Module, test_seq: torch.FloatTensor, test_lbl: torch.FloatTensor,
                   prefix: str, ds_lbl: str):
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
        # temp; will remove
        # test_seq = self.train_seq
        # test_lbl = self.train_lbls
        
        with torch.no_grad():
            for i in range(len(test_seq)):
                pred = model(test_seq[i])
                
                if self.device != 'cpu':
                    preds.append(pred.cpu())
                else:
                    preds.append(pred)
        

        plotting.plot_preds.plot_preds_from_device(preds, test_lbl, filename_prefix=prefix)

        unnormalized_preds, unnormalized_lbls = self.unnormalize_pred(preds, ds_lbl)
        error = MARE(unnormalized_preds, unnormalized_lbls)

        # metric computes but needs to be record
        


    
    def run_experiment(self, num_runs: int):
        model_labels = ['model_1', 'model_2', 'model_3', 'model_4']
        self.process_data()
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            for model_lbl in model_labels:
                cur_model = self.train_model(self.num_train_epochs, model_lbl)
                self.set_test_data(model_lbl)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f'{ds_lbl},{model_lbl},seed_{i}'
                    self.test_model(cur_model, test_seq, test_lbl, prefix)
        
                
    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        if self.device != 'cpu':
            if ds_lbl == 'dataset1':
                y_hat = preds.cpu().numpy() * self.first_piece_std + self.first_piece_mean
                unnormed_labels = self.labels1_tensor.cpu().numpy() * self.first_piece_std + self.first_piece_mean
            elif ds_lbl == 'dataset2':
                y_hat = preds.cpu().numpy() * self.second_piece_std + self.second_piece_mean
                unnormed_labels = self.labels2_tensor.cpu().numpy() * self.second_piece_std + self.second_piece_mean
            elif ds_lbl == 'dataset3':
                y_hat = preds.cpu().numpy() * self.third_piece_std + self.third_piece_mean
                unnormed_labels = self.labels3_tensor.cpu().numpy() * self.third_piece_std + self.third_piece_mean
            elif ds_lbl == 'dataset4':
                y_hat = preds.cpu().numpy() * self.fourth_piece_std + self.fourth_piece_mean
                unnormed_labels = self.labels4_tensor.cpu().numpy() * self.fourth_piece_std + self.fourth_piece_mean
            else:
                print('unnormalize_pred(): Incorrect dataset label')
                return False
            return y_hat, unnormed_labels
        else:
            if ds_lbl == 'dataset1':
                y_hat = preds.numpy() * self.first_piece_std + self.first_piece_mean
                unnormed_labels = self.labels1_tensor.numpy() * self.first_piece_std + self.first_piece_mean
            elif ds_lbl == 'dataset2':
                y_hat = preds.numpy() * self.second_piece_std + self.second_piece_mean
                unnormed_labels = self.labels2_tensor.numpy() * self.second_piece_std + self.second_piece_mean
            elif ds_lbl == 'dataset3':
                y_hat = preds.numpy() * self.third_piece_std + self.third_piece_mean
                unnormed_labels = self.labels3_tensor.numpy() * self.third_piece_std + self.third_piece_mean
            elif ds_lbl == 'dataset4':
                y_hat = preds.numpy() * self.fourth_piece_std + self.fourth_piece_mean
                unnormed_labels = self.labels4_tensor.numpy() * self.fourth_piece_std + self.fourth_piece_mean
            else:
                print('unnormalize_pred(): Incorrect dataset label')
                return False


    def set_test_data(self, model_lbl: str):
        if model_lbl == 'model_1':
            self.dataset_lbls = ['dataset2', 'dataset3', 'dataset4']
            self.test_seqs = [self.seq2_tensor, self.seq3_tensor, self.seq4_tensor]
            self.test_lbls = [self.labels2_tensor, self.labels3_tensor, self.labels4_tensor]
        elif model_lbl == 'model_2':
            self.dataset_lbls = ['dataset1', 'dataset3', 'dataset4']
            self.test_seqs = [self.seq1_tensor, self.seq3_tensor, self.seq4_tensor]
            self.test_lbls = [self.labels1_tensor, self.labels3_tensor, self.labels4_tensor]
        elif model_lbl == 'model_3':
            self.dataset_lbls = ['dataset1', 'dataset2', 'dataset4']
            self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq4_tensor]
            self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels4_tensor]
        elif model_lbl == 'model_4':
            self.dataset_lbls = ['dataset1', 'dataset2', 'dataset3']
            self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor]
            self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor]
        else:
            print('Wrong model label')
            return False