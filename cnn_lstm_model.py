import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel


LFLB_DEFAULT_CONFIG = [
    {
        'input_channels': 1,
        'output_channels': 64,
        'conv_kernel_size': 3,
        'pooling_kernel_size': 2,
        'pooling_stride': 2,
    },
    {
        'input_channels': 64,
        'output_channels': 64,
        'conv_kernel_size': 3,
        'pooling_kernel_size': [4, 2],
        'pooling_stride': [4, 2],
    },
    {
        'input_channels': 64,
        'output_channels': 128,
        'conv_kernel_size': 3,
        'pooling_kernel_size': [4, 2],
        'pooling_stride': [4, 2],
    },
    {
        'input_channels': 128,
        'output_channels': 128,
        'conv_kernel_size': 3,
        'pooling_kernel_size': [4, 2],
        'pooling_stride': [4, 2],
    },
]


class LocalFeatureLearningBlock2D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            conv_kernel_size,
            pooling_kernel_size,
            pooling_stride,
            dropout_rate=0.1,
            **kwargs):
        super(LocalFeatureLearningBlock2D, self).__init__()

        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            conv_kernel_size,
            padding='same',
            bias=False)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ELU()
        self.pooling = nn.MaxPool2d(pooling_kernel_size, pooling_stride)
        self.dropout = nn.Dropout2d(dropout_rate)

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')

    def reset_parameters(self):
        with torch.no_grad():
            self.conv.reset_parameters()
            self.batch_norm.reset_parameters()
            self.init_parameters()

    def forward(self, features):
        features = self.conv(features)
        features = self.batch_norm(features)
        features = self.activation(features)
        features = self.pooling(features)
        features = self.dropout(features)
        return features


class SelfAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads):
        super(SelfAttention, self).__init__()

        self.fc_hidden = nn.Linear(input_size, hidden_size, bias=False)
        self.activation = nn.ELU()
        self.fc_attention_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1, bias=False) for i in range(num_heads)])

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(
                self.fc_hidden.weight, nonlinearity='relu')

            for fc_attention_head in self.fc_attention_heads:
                nn.init.kaiming_uniform_(
                    fc_attention_head.weight, nonlinearity='linear')

    def reset_parameters(self):
        with torch.no_grad():
            self.fc_hidden.reset_parameters()
            for fc_attention_head in self.fc_attention_heads:
                fc_attention_head.reset_parameters()
            self.init_parameters()

    def forward(self, features):
        attention_features = self.fc_hidden(features)
        attention_features = self.activation(attention_features)

        attention_heads_scores = [
            F.softmax(fc_attention_head(attention_features), dim=0)
            for fc_attention_head in self.fc_attention_heads]

        features = [
            torch.sum(attention_scores * features, dim=0)
            for attention_scores in attention_heads_scores]

        features = torch.cat(features, dim=-1)
        return features


class CNNLSTM2DModel(BaseModel):
    def __init__(
            self,
            lflb_config=None,
            lstm_input_size=128,
            lstm_hidden_size=256,
            use_self_attention=False,
            self_attention_size=256,
            num_self_attention_heads=8,
            output_size=6,
            dropout_rate=0.1,
            label_smoothing=0.1,
            **kwargs):
        super(CNNLSTM2DModel, self).__init__()

        if lflb_config is None:
            lflb_config = LFLB_DEFAULT_CONFIG
        self.lflb_config = lflb_config
        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.use_self_attention = use_self_attention
        self.self_attention_size = self_attention_size
        self.num_self_attention_heads = num_self_attention_heads
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.label_smoothing = label_smoothing

        self.batch_norm_input = nn.BatchNorm2d(1)

        self.lflbs = nn.Sequential(*[
            LocalFeatureLearningBlock2D(
                **config, dropout_rate=self.dropout_rate)
            for config in self.lflb_config])

        self.lstm = nn.LSTM(
            self.lstm_input_size, self.lstm_hidden_size, bidirectional=True)

        if self.use_self_attention:
            self.self_attention = SelfAttention(
                self.lstm_hidden_size * 2,
                self.self_attention_size,
                self.num_self_attention_heads)

        fc_input_size = self.lstm_hidden_size * 2
        if self.use_self_attention:
            fc_input_size = fc_input_size * self.num_self_attention_heads
        self.fc = nn.Linear(fc_input_size, self.output_size)

        self.init_parameters()

    def config(self):
        config = {
            'lflb_config': self.lflb_config,
            'lstm_input_size': self.lstm_input_size,
            'lstm_hidden_size': self.lstm_hidden_size,
            'use_self_attention': self.use_self_attention,
            'self_attention_size': self.self_attention_size,
            'num_self_attention_heads': self.num_self_attention_heads,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'label_smoothing': self.label_smoothing,
        }
        return config

    def init_parameters(self):
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.fc.weight, nonlinearity='linear')
            nn.init.constant_(self.fc.bias, 0)

    def reset_parameters(self):
        with torch.no_grad():
            self.batch_norm_input.reset_parameters()

            for lflb in self.lflbs:
                lflb.reset_parameters()

            self.lstm.reset_parameters()

            if self.use_self_attention:
                self.self_attention.reset_parameters()

            self.fc.reset_parameters()

            self.init_parameters()

    def loss(self, features, labels):
        labels = torch.abs(labels - self.label_smoothing)
        loss = F.binary_cross_entropy_with_logits(features, labels)
        return loss

    def forward(self, features, labels=None, **kwargs):
        features = torch.unsqueeze(features, dim=1)
        features = self.batch_norm_input(features)

        features = self.lflbs(features)

        # Permute features into (length, batch, height, channels)
        features = torch.permute(features, (3, 0, 2, 1))
        features = torch.reshape(
            features, (features.shape[0], features.shape[1], -1))
        lstm_output, (hidden_state_n, cell_state_n) = self.lstm(features)

        if self.use_self_attention:
            features = self.self_attention(lstm_output)
        else:
            features = lstm_output[-1]

        features = self.fc(features)

        outputs = {
            'logits': features,
        }

        if labels is not None:
            outputs['loss'] = self.loss(features, labels)

        return outputs
