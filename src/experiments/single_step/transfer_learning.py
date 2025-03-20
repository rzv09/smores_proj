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
        self.seq1_tensor = sequences_3oec.make_sequence_tensor(self.train_seq1, self.device)
        self.labels1_tensor = sequences_3oec.make_sequence_tensor(self.train_labels1, self.device)
        self.seq2_tensor = sequences_3oec.make_sequence_tensor(self.train_seq2, self.device)
        self.labels2_tensor = sequences_3oec.make_sequence_tensor(self.train_labels2, self.device)
        self.seq3_tensor = sequences_3oec.make_sequence_tensor(self.train_seq3, self.device)
        self.labels3_tensor = sequences_3oec.make_sequence_tensor(self.train_labels3, self.device)
        self.seq4_tensor = sequences_3oec.make_sequence_tensor(self.train_seq4, self.device)
        self.labels4_tensor = sequences_3oec.make_sequence_tensor(self.train_labels4, self.device)
    
    def set_train_data(self, model_label: str):
        if model_label == 'model_123':
            self.train_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor]
            self.train_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor]
        elif model_label == 'model12':
            self.train_seqs = [self.seq1_tensor, self.seq2_tensor]
            self.train_lbls = [self.labels1_tensor, self.labels2_tensor]
        elif model_label == 'model124':
            self.train_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq4_tensor]
            self.train_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels4_tensor]
        else:
            print("set_train_data(): Model label does not exist")
            return False