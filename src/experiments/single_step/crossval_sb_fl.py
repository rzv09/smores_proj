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

class CrossValSBFL():
    def __init__(self, data_path_train, data_path_test, num_train_epochs, sampling_freq, sequence_len, device):
        self.data_path = data_path_train
        self.data_path_test = data_path_test
        self.num_train_epochs = num_train_epochs
        self.sampling_freq = sampling_freq
        self.sequence_len = sequence_len
        self.device = device

    def process_data(self):
        # preprocess TRAIN data and resample to sampling freq
        self.df = process_sb.make_timeseries(self.data_path)

        self.first_piece = process_sb.get_first_piece(self.df)
        self.first_piece_mean = self.first_piece.mean().values
        self.first_piece_std = self.first_piece.std().values
        self.second_piece = process_sb.get_second_piece(self.df)
        self.second_piece_mean = self.second_piece.mean().values
        self.second_piece_std = self.second_piece.std().values
        self.third_piece = process_sb.get_third_piece(self.df)
        self.third_piece_mean = self.third_piece.mean().values
        self.third_piece_std = self.third_piece.std().values
        self.fourth_piece = process_sb.get_fourth_piece(self.df)
        self.fourth_piece_mean = self.fourth_piece.mean().values
        self.fourth_piece_std = self.fourth_piece.std().values

        # preprocess TEST Florida data
        self.df_test = process_3oec.resample(process_3oec.make_timeseries(self.data_path_test), self.sampling_freq)
    
        self.first_piece_test = process_3oec.get_first_piece(self.df_test)
        self.first_piece_test_mean = self.first_piece_test.mean().values
        self.first_piece_test_std = self.first_piece_test.std().values
        self.second_piece_test = process_3oec.get_second_piece(self.df_test)
        self.second_piece_test_mean = self.second_piece_test.mean().values
        self.second_piece_test_std = self.second_piece_test.std().values
        self.third_piece_test = process_3oec.get_third_piece(self.df_test)
        self.third_piece_test_mean = self.third_piece_test.mean().values
        self.third_piece_test_std = self.third_piece_test.std().values
        self.fourth_piece_test = process_3oec.get_fourth_piece(self.df_test)
        self.fourth_piece_test_mean = self.fourth_piece_test.mean().values
        self.fourth_piece_test_std = self.fourth_piece_test.std().values


        # create train sequences
        self.train_seq1, self.train_labels1 = create_sequences_single_step(standardize_piece(self.first_piece).values, self.sequence_len)
        self.train_seq2, self.train_labels2 = create_sequences_single_step(standardize_piece(self.second_piece).values, self.sequence_len)
        self.train_seq3, self.train_labels3 = create_sequences_single_step(standardize_piece(self.third_piece).values, self.sequence_len)
        self.train_seq4, self.train_labels4 = create_sequences_single_step(standardize_piece(self.fourth_piece).values, self.sequence_len)

        # create TEST (fl=Florida) sequences
        self.fl_seq1, self.fl_labels1 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.first_piece_test).values,
                                                                       self.sequence_len)
        self.fl_seq2, self.fl_labels2 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.second_piece_test).values,
                                                                      self.sequence_len)
        self.fl_seq3, self.fl_labels3 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.third_piece_test).values,
                                                                      self.sequence_len)
        self.fl_seq4, self.fl_labels4 = sequences_3oec.create_sequences_single_step(process_3oec.standardize_piece(self.fourth_piece_test).values,
                                                                      self.sequence_len)
        
        # create train tensors
        self.seq1_tensor = make_sequence_tensor(self.train_seq1, self.device)
        self.labels1_tensor = make_sequence_tensor(self.train_labels1, self.device)
        self.seq2_tensor = make_sequence_tensor(self.train_seq2, self.device)
        self.labels2_tensor = make_sequence_tensor(self.train_labels2, self.device)
        self.seq3_tensor = make_sequence_tensor(self.train_seq3, self.device)
        self.labels3_tensor = make_sequence_tensor(self.train_labels3, self.device)
        self.seq4_tensor = make_sequence_tensor(self.train_seq4, self.device)
        self.labels4_tensor = make_sequence_tensor(self.train_labels4, self.device)

        # create test tensors
        self.fl1_tensor = make_sequence_tensor(self.fl_seq1, self.device)
        self.fl_labels1_tensor = make_sequence_tensor(self.fl_labels1, self.device)
        self.fl2_tensor = make_sequence_tensor(self.fl_seq2, self.device)
        self.fl_labels2_tensor = make_sequence_tensor(self.fl_labels2, self.device)
        self.fl3_tensor = make_sequence_tensor(self.fl_seq3, self.device)
        self.fl_labels3_tensor = make_sequence_tensor(self.fl_labels3, self.device)
        self.fl4_tensor = make_sequence_tensor(self.fl_seq4, self.device)
        self.fl_labels4_tensor = make_sequence_tensor(self.fl_labels4, self.device)

        
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
        """
        self.set_train_data(model_label)

        model = lstm_single_step.create_lstm_single_step()
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.SmoothL1Loss()

        train_losses = []

        for epoch in range(num_epochs):
            for k in range(len(self.train_seq)):
                model.train()
                optimizer.zero_grad()
                pred = model(self.train_seq[k])

                loss = criterion(pred, self.train_lbls[k])
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
        self.set_train_data(model_label)

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
            for k in range(len(self.train_seq)):
                optimizer.zero_grad()
                pred = model(self.train_seq[k])
                loss = criterion(pred, self.train_lbls[k])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
            avg_train_loss = epoch_loss / len(self.train_seq)
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


        # criterion2(torch.FloatTensor(y_pred).to(device=device), test_tensor_labels_2.squeeze(1)).item()

        l1_smooth_error = nn.SmoothL1Loss()
        l1_smooth_value = l1_smooth_error(torch.FloatTensor(preds).to(device='cpu'), test_lbl.squeeze(1).to(device='cpu')).item()

        unnormalized_preds, unnormalized_lbls = self.unnormalize(preds, ds_lbl)
        error = MARE(unnormalized_preds, unnormalized_lbls)
        
        csv_filepath = write_csv(model_lbl, ds_lbl, seed, error.item(), l1_smooth_error, 
                                 l1_smooth_value, train_losses, save_dir)
        return csv_filepath


    
    def run_experiment(self, num_runs: int, save_dir : str ='./out/exp1', early_stop: bool = False):
        os.mkdir(save_dir)
        l1_smooth_error = nn.SmoothL1Loss()
        model_labels = ['model_1', 'model_2', 'model_3', 'model_4']
        self.process_data()
        for i in range(num_runs):
            utils.set_seeds.set_experiment_seeds(i)
            for model_lbl in model_labels:
                if early_stop:
                    cur_model, train_losses = self.train_model_es(self.num_train_epochs, model_lbl)
                else:
                    cur_model, train_losses = self.train_model(self.num_train_epochs, model_lbl)
                self.set_test_data(model_lbl)
                for test_seq, test_lbl, ds_lbl in zip(self.test_seqs, self.test_lbls, self.dataset_lbls):
                    prefix = f'{model_lbl},seed_{i},{ds_lbl}'
                    csv_filepath = self.test_model(cur_model, model_lbl, i, test_seq, test_lbl, prefix, ds_lbl, train_losses, save_dir)
        
        write_stats_mare(csv_filepath, save_dir)
        write_stats_crit(csv_filepath, save_dir, l1_smooth_error)
        plot_MARE(csv_filepath, save_dir)
        plot_smooth_l1_loss(csv_filepath, save_dir)

                
    def unnormalize(self, preds: torch.FloatTensor, ds_lbl: str):
        preds = torch.FloatTensor(preds)
        preds = preds.cpu()
        if ds_lbl == 'florida_dataset1':
            y_hat = preds.numpy() * self.first_piece_test_std + self.first_piece_test_mean
            unnormed_labels = self.fl_labels1_tensor.squeeze(1).cpu().numpy() * self.first_piece_test_std + self.first_piece_test_mean
        elif ds_lbl == 'florida_dataset2':
            y_hat = preds.numpy() * self.second_piece_test_std + self.second_piece_test_mean
            unnormed_labels = self.fl_labels2_tensor.squeeze(1).cpu().numpy() * self.second_piece_test_std + self.second_piece_test_mean
        elif ds_lbl == 'florida_dataset3':
            y_hat = preds.numpy() * self.third_piece_test_std + self.third_piece_test_mean
            unnormed_labels = self.fl_labels3_tensor.squeeze(1).cpu().numpy() * self.third_piece_test_std + self.third_piece_test_mean
        elif ds_lbl == 'florida_dataset4':
            y_hat = preds.numpy() * self.fourth_piece_test_std + self.fourth_piece_test_mean
            unnormed_labels = self.fl_labels4_tensor.squeeze(1).cpu().numpy() * self.fourth_piece_test_std + self.fourth_piece_test_mean
        else:
            print('unnormalize_pred(): Incorrect dataset label')
            return False
        return y_hat, unnormed_labels


    def set_test_data(self, model_lbl: str, test_only: bool=False):
        # if not test_only:
        #     self.dataset_lbls = ['dataset1', 'dataset2', 'dataset3', 'dataset4']
        #     self.test_seqs = [self.seq1_tensor, self.seq2_tensor, self.seq3_tensor, self.seq4_tensor]
        #     self.test_lbls = [self.labels1_tensor, self.labels2_tensor, self.labels3_tensor, self.labels4_tensor]
        #     return True
        self.dataset_lbls = ['florida_dataset1', 'florida_dataset2', 'florida_dataset3', 'florida_dataset4']
        self.test_seqs = [self.fl1_tensor, self.fl2_tensor, self.fl3_tensor, self.fl4_tensor]
        self.test_lbls = [self.fl_labels1_tensor, self.fl_labels2_tensor, self.fl_labels3_tensor, self.fl_labels4_tensor]
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