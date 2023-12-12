import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import os
import copy
import pandas as pd
from torch.nn.modules import TransformerEncoderLayer

class FeatureRegression(nn.Module):
    def __init__(self, input_size):
        super(FeatureRegression, self).__init__()
        self.build(input_size)

    def build(self, input_size):
        self.W = Parameter(torch.Tensor(input_size, input_size))
        self.b = Parameter(torch.Tensor(input_size))
        m = torch.ones(input_size, input_size) - torch.eye(input_size, input_size)
        self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, x):
        z_h = F.linear(x, self.W * Variable(self.m), self.b)
        return z_h

class Decay(nn.Module):
    def __init__(self, input_size, output_size, diag=False):
        super(Decay, self).__init__()
        self.diag = diag
        self.build(input_size, output_size)

    def build(self, input_size, output_size):
        self.W = Parameter(torch.Tensor(output_size, input_size))
        self.b = Parameter(torch.Tensor(output_size))

        if self.diag == True:
            assert(input_size == output_size)
            m = torch.eye(input_size, input_size)
            self.register_buffer('m', m)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(0))
        self.W.data.uniform_(-stdv, stdv)
        if self.b is not None:
            self.b.data.uniform_(-stdv, stdv)

    def forward(self, d):
        if self.diag == True:
            gamma = F.relu(F.linear(d, self.W * Variable(self.m), self.b))
        else:
            gamma = F.relu(F.linear(d, self.W, self.b))
        gamma = torch.exp(-gamma)
        return gamma

class Decay_obs(nn.Module):
    def __init__(self, input_size, output_size):
        super(Decay_obs, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, delta_diff):
        # When delta_diff is negative, weight tends to 1.
        # When delta_diff is positive, weight tends to 0.
        sign = torch.sign(delta_diff)
        weight_diff = self.linear(delta_diff)
        # weight_diff can be either positive or negative for each delta_diff
        positive_part = F.relu(weight_diff)
        negative_part = F.relu(-weight_diff)
        weight_diff = positive_part + negative_part
        weight_diff = sign * weight_diff
        # Using a tanh activation to squeeze values between -1 and 1
        weight_diff = torch.tanh(weight_diff)
        # This will move the weight values towards 1 if delta_diff is negative 
        # and towards 0 if delta_diff is positive
        weight = 0.5 * (1 - weight_diff)

        return weight

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class rits(nn.Module):
    def __init__(self, args, dropout=0.25):
        super(rits, self).__init__()
        self.args = args
        # Define Input Size Depends on the Dataset
        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'physionet_all':
            input_size = 35
        elif self.args.dataset == 'mimic_89f':
            input_size = 89
        elif self.args.dataset == 'mimic_59f':
            input_size = 59
        elif self.args.dataset == 'eicu':
            input_size = 20
        elif self.args.dataset == 'air':
            input_size = 18
        elif self.args.dataset == 'traffic':
            input_size = 58

        self.input_size = input_size
        self.hidden_size = self.args.hiddens
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, mask, deltas, h=None, get_y=False):
        # Get dimensionality
        [B, T, V] = x.shape
        
        if h == None:
            h = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        for t in range(T):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h
            
            # history based estimation
            x_h = self.hist(h)
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # feature based estimation
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)
            
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)

            h = self.gru(input_t, h)

            # Keep the imputation
            Hiddens.append(h.unsqueeze(dim=1))
        Hiddens = torch.cat(Hiddens, dim=1)

        if (self.args.task in ['C', 'pretrain', 'pretrain_brits', 'pretrain_train', 'pretrain_brits_freeze']) and (get_y == True):
            y_out = self.classification(self.dropout(h))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        ret = {'imputation':x_imp, 'xloss':x_loss, 'hidden_state':Hiddens, 'y_out':y_out, 'y_score':y_score}

        return ret

class brits(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(brits, self).__init__()
        self.args = args
        self.model_f = rits(args=self.args)
        self.model_b = rits(args=self.args)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)

        ret_f = self.model_f(x, m, d_f, get_y=self.get_y)
        # Set data to be backward
        x_b = x.flip(dims=[1])
        m_b = m.flip(dims=[1])
        ret_b = self.model_b(x_b, m_b, d_b, get_y=self.get_y)

        # Averaging the imputations and prediction
        x_imp = (ret_f['imputation'] + ret_b['imputation'].flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(ret_f['imputation'] - ret_b['imputation'].flip(dims=[1])).mean() * 1e-1

        # average the regression loss
        xreg_loss = ret_f['xloss'] + ret_b['xloss']

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'y_out_f':ret_f['y_out'], 'y_score_f':ret_f['y_score'], 'y_out_b':ret_b['y_out'], 'y_score_b':ret_b['y_score']}
        return ret

class csai(nn.Module):
    def __init__(self, args, dropout=0.25, medians_df=None):
        super(csai, self).__init__()
        self.args = args

        if medians_df is not None:
            self.medians_tensor = torch.tensor(list(medians_df.values())).float().to(self.args.device)
        else:
            self.medians_tensor = None

        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'physionet_all':
            input_size = 35
        elif self.args.dataset == 'mimic_89f':
            input_size = 89
        elif self.args.dataset == 'mimic_59f':
            input_size = 59
        elif self.args.dataset == 'eicu':
            input_size = 20
        elif self.args.dataset == 'air':
            input_size = 18
        elif self.args.dataset == 'traffic':
            input_size = 58

        self.step_channels = self.args.step_channels

        self.input_size = input_size
        self.hidden_size = self.args.hiddens
        self.temp_decay_h = Decay(input_size=self.input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=self.input_size, output_size=self.input_size, diag = True)
        self.hist = nn.Linear(self.hidden_size, self.input_size)
        self.feat_reg_v = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.weighted_obs = Decay_obs(self.input_size, self.input_size)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        
        self.pos_encoder = PositionalEncoding(self.step_channels)
        self.input_projection = Conv1d_with_init(self.input_size, self.step_channels, 1)
        self.output_projection1 = Conv1d_with_init(self.step_channels, self.hidden_size, 1)
        self.output_projection2 = Conv1d_with_init(self.args.hours*2, 1, 1)
        self.time_layer = get_torch_trans(channels=self.step_channels)

        self.reset_parameters()
    
    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, mask, deltas, last_obs, h=None, get_y=True):
        # Get dimensionality
        [B, T, _] = x.shape

        if self.medians_tensor is not None:
            medians_t = self.medians_tensor.unsqueeze(0).repeat(B, 1)

        decay_factor = self.weighted_obs(deltas - medians_t.unsqueeze(1))

        if h == None:
            data_last_obs = self.input_projection(last_obs.permute(0, 2, 1)).permute(0, 2, 1)
            data_decay_factor = self.input_projection(decay_factor.permute(0, 2, 1)).permute(0, 2, 1)

            data_last_obs = self.pos_encoder(data_last_obs.permute(1, 0, 2)).permute(1, 0, 2)
            data_decay_factor = self.pos_encoder(data_decay_factor.permute(1, 0, 2)).permute(1, 0, 2)
            
            data = torch.cat([data_last_obs, data_decay_factor], dim=1)

            data = self.time_layer(data)
            data = self.output_projection1(data.permute(0, 2, 1)).permute(0, 2, 1)
            h = self.output_projection2(data).squeeze()
            # h = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        x_loss = 0
        x_imp = x.clone()
        Hiddens = []
        for t in range(T):
            x_t = x[:, t, :]
            d_t = deltas[:, t, :]
            m_t = mask[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h
            
            # history based estimation
            x_h = self.hist(h)        
            
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            # feature based estimation
            xu = self.feat_reg_v(x_r_t)
            gamma_x = self.temp_decay_x(d_t)
            
            beta = self.weight_combine(torch.cat([gamma_x, m_t], dim=1))
            x_comb_t = beta * xu + (1 - beta) * x_h

            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp[:, t, :] = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the RNN
            input_t = torch.cat([x_imp[:, t, :], m_t], dim=1)

            h = self.gru(input_t, h)
            Hiddens.append(h.unsqueeze(dim=1))

        Hiddens = torch.cat(Hiddens, dim=1)

        if (self.args.task in ['C', 'pretrain', 'pretrain_brits', 'pretrain_train', 'pretrain_brits_freeze']) and (get_y == True):
            y_out = self.classification(self.dropout(h))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        ret = {'imputation':x_imp, 'xloss':x_loss, 'hidden_state':Hiddens, 'y_out':y_out, 'y_score':y_score}

        return ret

class bcsai(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(bcsai, self).__init__()
        self.args = args
        self.model_f = csai(args=self.args, medians_df=medians_df)
        self.model_b = csai(args=self.args, medians_df=medians_df)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)
        last_obs_f = xdata['last_obs_f'].to(self.args.device)
        last_obs_b = xdata['last_obs_b'].to(self.args.device)

        ret_f = self.model_f(x, m, d_f, last_obs_f, get_y=self.get_y)
        # Set data to be backward
        x_b = x.flip(dims=[1])
        m_b = m.flip(dims=[1])
        ret_b = self.model_b(x_b, m_b, d_b, last_obs_b, get_y=self.get_y)

        # Averaging the imputations and prediction
        x_imp = (ret_f['imputation'] + ret_b['imputation'].flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(ret_f['imputation'] - ret_b['imputation'].flip(dims=[1])).mean() * 1e-1

        # average the regression loss
        xreg_loss = ret_f['xloss'] + ret_b['xloss']

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'y_out_f':ret_f['y_out'], 'y_score_f':ret_f['y_score'], 'y_out_b':ret_b['y_out'], 'y_score_b':ret_b['y_score']}
        return ret

class gru_d(nn.Module):
    def __init__(self, args, dropout=0.25, medians_df=None, get_y=False):
        super(gru_d, self).__init__()
        self.args = args
        # Define Input Size Depends on the Dataset
        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'mimic_89f':
            input_size = 89
        elif self.args.dataset == 'mimic_59f':
            input_size = 59
        elif self.args.dataset == 'eicu':
            input_size = 20
        elif self.args.dataset == 'air':
            input_size = 18
        elif self.args.dataset == 'traffic':
            input_size = 58

        self.get_y = get_y
        self.input_size = input_size
        self.hidden_size = self.args.hiddens
        self.temp_decay_h = Decay(input_size=input_size, output_size=self.hidden_size, diag = False)
        self.temp_decay_x = Decay(input_size=input_size, output_size=self.input_size, diag = True)
        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, xdata, meanset, direct='forward', hidden=None):
        x = xdata['values'].to(self.args.device)
        mask = xdata['masks'].to(self.args.device)
        if direct=='forward':
            deltas = xdata['deltas_f'].to(self.args.device)
        elif direct=='backward':
            x = x.flip(dims=[1])
            mask = mask.flip(dims=[1])
            deltas = xdata['deltas_b'].to(self.args.device)

        meanset = torch.tensor(meanset).to(self.args.device)
        
        x_original = copy.deepcopy(x)
        x_original[mask==0] = np.nan
        x_forward = [pd.DataFrame(x_original[i,:,:].cpu().numpy()).fillna(method='ffill').fillna(0.0).values for i in range(x_original.size(0))]
        x_forward = torch.from_numpy(np.array(x_forward)).to(self.args.device)

        [B, T, V] = x.shape

        if hidden == None:
            hidden = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)
        
        x_loss = 0
        x_imp = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = mask[:, t, :]
            d_t = deltas[:, t, :]
            f_t = x_forward[:, t, :]

            gamma_h = self.temp_decay_h(d_t)
            hidden = hidden * gamma_h

            gamma_x = self.temp_decay_x(d_t)
            x_u = gamma_x * f_t + (1 - gamma_x) *  meanset

            x_loss += torch.sum(torch.abs(x_t - x_u) * m_t) / (torch.sum(m_t) + 1e-5)
            
            x_h = m_t * x_t + (1 - m_t) * x_u
            inputs = torch.cat([x_h, m_t], dim = 1).float()

            hidden = self.gru(inputs, hidden)

            x_imp.append(x_h.unsqueeze(dim = 1))

        x_imp = torch.cat(x_imp, dim = 1)

        if (self.args.task in ['C', 'pretrain', 'pretrain_brits', 'pretrain_train',]) and (self.get_y == True):
            y_out = self.classification(self.dropout(hidden))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        ret = {'imputation':x_imp, 'loss_consistency':0, 'loss_regression':x_loss, 'y_out_f':y_out, 'y_score_f':y_score, 'y_out_b':y_out, 'y_score_b':y_score}
        return ret

class m_rnn(nn.Module):
    def __init__(self, args, dropout=0.25, medians_df=None, get_y=False):
        super(m_rnn, self).__init__()
        self.args = args
        # Define Input Size Depends on the Dataset
        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'mimic_89f':
            input_size = 89
        elif self.args.dataset == 'mimic_59f':
            input_size = 59
        elif self.args.dataset == 'eicu':
            input_size = 20
        elif self.args.dataset == 'air':
            input_size = 18
        elif self.args.dataset == 'traffic':
            input_size = 58

        self.input_size = input_size
        self.hidden_size = self.args.hiddens
        self.get_y = get_y
        self.hist_reg = nn.Linear(self.hidden_size * 2, self.input_size)
        self.feat_reg = FeatureRegression(self.input_size)
        self.weight_combine = nn.Linear(self.input_size * 2, self.input_size)
        self.imputation = nn.Linear(self.input_size, self.input_size)

        self.rnn_cell = nn.GRUCell(self.input_size * 3, self.hidden_size)
        self.pred_rnn = nn.GRU(self.input_size, self.hidden_size, batch_first = True)

        self.dropout = nn.Dropout(dropout)
        self.classification = nn.Linear(self.hidden_size, self.args.out_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def get_hidden(self, xdata, direct, hidden=None):
        x = xdata['values'].to(self.args.device)
        masks = xdata['masks'].to(self.args.device)
        if direct=='forward':
            deltas = xdata['deltas_f'].to(self.args.device)
        elif direct=='backward':
            x = x.flip(dims=[1])
            masks = masks.flip(dims=[1])
            deltas = xdata['deltas_b'].to(self.args.device)
        [B, T, V] = x.shape
        hiddens = []
        if hidden == None:
            hidden = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        for t in range(T):
            hiddens.append(hidden)
            x_t = x[:, t, :]
            m_t = masks[:, t, :]
            d_t = deltas[:, t, :]
            inputs = torch.cat([x_t, m_t, d_t], dim = 1)
            hidden = self.rnn_cell(inputs, hidden)
        return hiddens

    def forward(self, xdata, direct='forward'):

        hidden_forward = self.get_hidden(xdata, 'forward')
        hidden_backward = self.get_hidden(xdata, 'backward')[::-1]

        x = xdata['values'].to(self.args.device)
        masks = xdata['masks'].to(self.args.device)
        if direct=='forward':
            deltas = xdata['deltas_f'].to(self.args.device)
        elif direct=='backward':
            x = x.flip(dims=[1])
            masks = masks.flip(dims=[1])
            deltas = xdata['deltas_b'].to(self.args.device)

        [B, T, V] = x.shape
        x_loss = 0
        x_imp = []

        for t in range(T):
            x_t = x[:, t, :]
            m_t = masks[:, t, :]
            d_t = deltas[:, t, :]

            hf = hidden_forward[t]
            hb = hidden_backward[t]
            h = torch.cat([hf, hb], dim = 1)

            x_v = self.hist_reg(h)
            x_u = self.feat_reg(x_t)
            x_h = x_u + self.weight_combine(torch.cat([x_v, m_t], dim = 1))
            x_imp_t = self.imputation(x_h)

            x_loss += torch.sum(torch.abs(x_t - x_imp_t) * m_t) / (torch.sum(m_t) + 1e-5)

            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_imp_t)
            x_imp.append(x_imp_t.unsqueeze(dim = 1))

        x_imp = torch.cat(x_imp, dim = 1)

        if (self.args.task in ['C', 'pretrain', 'pretrain_brits', 'pretrain_train',]) and (self.get_y == True):
            out, h = self.pred_rnn(x_imp)
            y_out = self.classification(self.dropout(h.squeeze()))
            y_score = torch.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0
        
        ret = {'imputation':x_imp, 'loss_consistency':0, 'loss_regression':x_loss, 'y_out_f':y_out, 'y_score_f':y_score, 'y_out_b':y_out, 'y_score_b':y_score}
        return ret

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.args = args
        self.hiddens = self.args.vae_hiddens

        # Encoder
        self.enc = nn.Sequential()
        for i in range(len(self.hiddens)-2):
            self.enc.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i+1]))
            self.enc.add_module("bn_%d" % i, nn.BatchNorm1d(self.hiddens[i+1]))
            self.enc.add_module("do_%d" % i, nn.Dropout(self.args.keep_prob))
            self.enc.add_module("tanh_%d" % i, nn.Tanh())
        self.enc_mu = nn.Linear(self.hiddens[-2], self.hiddens[-1])
        self.enc_logvar = nn.Linear(self.hiddens[-2], self.hiddens[-1])

        # Decoder
        self.dec = nn.Sequential()
        for i in range(len(self.hiddens))[::-1][:-2]:
            self.dec.add_module("fc_%d" % i, nn.Linear(self.hiddens[i], self.hiddens[i-1]))
            self.dec.add_module("bn_%d" % i, nn.BatchNorm1d(self.hiddens[i-1]))
            self.dec.add_module("do_%d" % i, nn.Dropout(self.args.keep_prob))
            self.dec.add_module("tanh_%d" % i, nn.Tanh())
        self.dec_mu = nn.Linear(self.hiddens[1], self.hiddens[0])
        self.dec_logvar = nn.Linear(self.hiddens[1], self.hiddens[0])

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    # Reparameterize
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):

        # Encoding
        e = self.enc(x)
        enc_mu = self.enc_mu(e)
        enc_logvar =self.enc_logvar(e)
        z = self.reparameterize(enc_mu, enc_logvar)

        # Decoding
        d = self.dec(z)
        dec_mu = self.dec_mu(d)
        dec_logvar = self.dec_logvar(d)
        x_hat = dec_mu

        return z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar

class RIN(nn.Module):
    def __init__(self, args):#
        super(RIN, self).__init__()
        self.args = args
        # Define Input Size Depends on the Dataset
        if self.args.dataset == 'physionet':
            input_size = 35
        elif self.args.dataset == 'mimic_89f':
            input_size = 89
        elif self.args.dataset == 'mimic_59f':
            input_size = 59
        elif self.args.dataset == 'eicu':
            input_size = 20
        elif self.args.dataset == 'air':
            input_size = 18
        elif self.args.dataset == 'traffic':
            input_size = 58
            
        self.input_size = input_size
        self.hidden_size = self.args.hiddens

        self.hist = nn.Linear(self.hidden_size, input_size)
        self.conv1 = nn.Conv1d(2, 1, kernel_size=1, stride=1)
        self.temp_decay_h = Decay(input_size=input_size, output_size=self.hidden_size)
        self.feat_reg_v = FeatureRegression(input_size)
        self.feat_reg_r = FeatureRegression(input_size)

        self.unc_flag = self.args.unc_flag
        self.gru = nn.GRUCell(self.input_size * 2, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        # Activate only for the model with uncertainty
        if self.args.unc_flag == 1:
            self.unc_decay = Decay(input_size=input_size, output_size=input_size)

        self.reset_parameters()

    def reset_parameters(self):
        for weight in self.parameters():
            if len(weight.size()) == 1:
                continue
            stv = 1. / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stv, stv)

    def forward(self, x, x_hat, u, m, d, h=None, get_y=False):
        # Get dimensionality
        [B, T, _] = x.shape

        # Initialize Hidden weights
        if h == None:
            h = Variable(torch.zeros(B, self.hidden_size)).to(self.args.device)

        x_loss = 0
        # x_imp = torch.Tensor().cuda()
        x_imp = []
        xus = []
        xrs = []
        for t in range(T):
            x_t = x[:, t, :]
            x_hat_t = x_hat[:, t, :]
            u_t = u[:, t, :]
            d_t = d[:, t, :]
            m_t = m[:, t, :]

            # Decayed Hidden States
            gamma_h = self.temp_decay_h(d_t)
            h = h * gamma_h

            # Regression
            x_h = self.hist(h)
            x_r_t = (m_t * x_t) + ((1 - m_t) * x_h)

            if self.args.unc_flag == 1:
                xbar = (m_t * x_t) + ((1 - m_t) * x_hat_t)
                xu = self.feat_reg_v(xbar) * self.unc_decay(u_t)
            else:
                xbar = (m_t * x_t) + ((1 - m_t) * x_hat_t)
                xu = self.feat_reg_v(xbar)

            xr = self.feat_reg_r(x_r_t)

            x_comb_t = self.conv1(torch.cat([xu.unsqueeze(1), xr.unsqueeze(1)], dim=1)).squeeze(1)
            x_loss += torch.sum(torch.abs(x_t - x_comb_t) * m_t) / (torch.sum(m_t) + 1e-5)

            # Final Imputation Estimates
            x_imp_t = (m_t * x_t) + ((1 - m_t) * x_comb_t)

            # Set input the the RNN
            input_t = torch.cat([x_imp_t, m_t], dim=1)

            # Feed into GRU cell, get the hiddens
            h = self.gru(input_t, h)

            # Keep the imputation
            x_imp.append(x_imp_t.unsqueeze(dim=1))
            xus.append(xu.unsqueeze(dim=1))
            xrs.append(xr.unsqueeze(dim=1))

        x_imp = torch.cat(x_imp, dim=1)
        xus = torch.cat(xus, dim=1)
        xrs = torch.cat(xrs, dim=1)

        # Get the output
        if (self.args.task in ['C', 'pretrain', 'pretrain_brits', 'pretrain_train',]) and (get_y == True):
            y_out = self.fc_out(h)
            y_score = self.sigmoid(y_out)
        else:
            y_out = 0
            y_score = 0

        return x_imp, y_out, y_score, x_loss, xus, xrs

class bvrin(nn.Module):
    def __init__(self, args, medians_df=None, get_y=False):
        super(bvrin, self).__init__()
        self.args = args
        self.vae = VAE(self.args)
        self.rin_f = RIN(self.args)
        self.rin_b = RIN(self.args)
        self.criterion_vae = SVAELoss(self.args)
        self.get_y = get_y

    def forward(self, xdata):
        x = xdata['values'].to(self.args.device)
        m = xdata['masks'].to(self.args.device)
        d_f = xdata['deltas_f'].to(self.args.device)
        d_b = xdata['deltas_b'].to(self.args.device)
        eval_x = xdata['evals'].to(self.args.device)
        eval_m = xdata['eval_masks'].to(self.args.device)
        y = xdata['labels'].to(self.args.device)
        
        [B, T, V] = x.shape
        # VAE
        rx = x.contiguous().view(-1, V)
        rm = m.contiguous().view(-1, V)
        z, enc_mu, enc_logvar, x_hat, dec_mu, dec_logvar = self.vae(rx)
        unc = (m * torch.zeros(B, T, V).to(self.args.device)) + ((1 - m) * torch.exp(0.5 * dec_logvar).view(B, T, V))

        # RIN Forward
        x_imp_f, y_out_f, y_score_f, xreg_loss_f, _, _ = self.rin_f(x, x_hat.view(B, T, V), unc, m, d_f, get_y=self.get_y)

        # Set data to be backward
        x_b = x.flip(dims=[1])
        x_hat_b = x_hat.view(B, T, V).flip(dims=[1])
        unc_b = unc.flip(dims=[1])
        m_b = m.flip(dims=[1])

        # RIN Backward
        x_imp_b, y_out_b, y_score_b, xreg_loss_b, _, _ = self.rin_b(x_b, x_hat_b, unc_b, m_b, d_b, get_y=self.get_y)

        loss_vae, lossnll, lossmae, losskld, lossl1 = self.criterion_vae(self.vae, rx, eval_x.view(B*T, V), x_hat.view(B*T, V), rm, eval_m.view(B*T, V), enc_mu, enc_logvar, dec_mu, dec_logvar, phase='train')   

        # Averaging the imputations and prediction
        x_imp = (x_imp_f + x_imp_b.flip(dims=[1])) / 2
        x_imp = (x * m)+ ((1-m) * x_imp)

        # Add consistency loss
        loss_consistency = torch.abs(x_imp_f - x_imp_b.flip(dims=[1])).mean() * 1e-1

        # Sum the regression loss
        xreg_loss = xreg_loss_f + xreg_loss_b

        ret = {'imputation':x_imp, 'loss_consistency':loss_consistency, 'loss_regression':xreg_loss, 'loss_vae':loss_vae, 'y_out_f':y_out_f, 'y_score_f':y_score_f, 'y_out_b':y_out_b, 'y_score_b':y_score_b}
        return ret