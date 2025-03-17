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
class CrossVal():
    def __init__(self, data_path, sampling_freq, sequence_len, device):
        self.data_path = data_path
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device

    def process_data(self):
        # preprocess data and resample to sampling freq
        self.df = process_3oec.resample(process_3oec.make_timeseries(self.data_path), self.sampling_freq)

        self.first_piece = process_3oec.get_first_piece(self.df)
        self.second_piece = process_3oec.get_second_piece(self.df)
        self.third_piece = process_3oec.get_third_piece(self.df)
        self.fourth_piece = process_3oec.get_fourth_piece(self.df)


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
        self.process_data()
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
    
    def test_model(self, model: nn.Module, model_lbl: str):
        """
        Test a trained model

        Args:
            model (nn.Module)
            model_lbl (str)
            test_seq (torch.FloatTensor)
            test_lbl (torch.FloatTensor)
        """
        preds = []
        model.eval()
        # temp; will remove
        test_seq = self.train_seq
        test_lbl = self.train_lbls
        
        with torch.no_grad():
            for i in range(len(test_seq)):
                pred = model(test_seq[i])
                
                if self.device != 'cpu':
                    preds.append(pred.cpu())
                else:
                    preds.append(pred)
        
        plotting.plot_preds.plot_preds_from_device(preds, test_lbl)

        ## add metrics reporting later



