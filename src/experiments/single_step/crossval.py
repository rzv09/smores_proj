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

class CrossVal():
    def __init__(self, data_path, sampling_freq, sequence_len):
        # self.model = lstm_single_step()
        self.data_path = data_path
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len

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
        self.train_seq1_tensor = sequences_3oec.make_sequence_tensor(self.train_seq1)
        self.train_labels1_tensor = sequences_3oec.make_sequence_tensor(self.train_labels1)
        self.train_seq2_tensor = sequences_3oec.make_sequence_tensor(self.train_seq2)
        self.train_labels2_tensor = sequences_3oec.make_sequence_tensor(self.train_labels2)
        self.train_seq3_tensor = sequences_3oec.make_sequence_tensor(self.train_seq3)
        self.train_labels3_tensor = sequences_3oec.make_sequence_tensor(self.train_labels3)
        self.train_seq4_tensor = sequences_3oec.make_sequence_tensor(self.train_seq4)
        self.train_labels4_tensor = sequences_3oec.make_sequence_tensor(self.train_labels4)
        

