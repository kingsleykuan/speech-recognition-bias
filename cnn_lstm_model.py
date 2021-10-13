import torch
import torch.nn as nn
import torch.nn.functional as F

from base_model import BaseModel


LFLB_DEFAULT_CONFIG = [
    {
        'input_channels': 1,
        'output_channels': 64,
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pooling_kernel_size': 2,
        'pooling_stride': 2,
        'dropout_rate': 0.1,
    },
    {
        'input_channels': 64,
        'output_channels': 64,
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pooling_kernel_size': 4,
        'pooling_stride': 4,
        'dropout_rate': 0.1,
    },
    {
        'input_channels': 64,
        'output_channels': 128,
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pooling_kernel_size': 4,
        'pooling_stride': 4,
        'dropout_rate': 0.1,
    },
    {
        'input_channels': 128,
        'output_channels': 128,
        'conv_kernel_size': 3,
        'conv_stride': 1,
        'pooling_kernel_size': 4,
        'pooling_stride': 4,
        'dropout_rate': 0.1,
    },
]


class LocalFeatureLearningBlock2D(nn.Module):
    def __init__(
            self,
            input_channels,
            output_channels,
            conv_kernel_size,
            conv_stride,
            pooling_kernel_size,
            pooling_stride,
            dropout_rate=0.1,
            **kwargs):
        super(LocalFeatureLearningBlock2D, self).__init__()

        self.conv = nn.Conv2d(
            input_channels,
            output_channels,
            conv_kernel_size,
            conv_stride,
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


class CNNLSTM2DModel(BaseModel):
    def __init__(
            self,
            lflb_config=None,
            lstm_input_size=128,
            lstm_output_size=256,
            output_size=6,
            label_smoothing=0.1,
            **kwargs):
        super(CNNLSTM2DModel, self).__init__()

        if lflb_config is None:
            lflb_config = LFLB_DEFAULT_CONFIG
        self.lflb_config = lflb_config
        self.lstm_input_size = lstm_input_size
        self.lstm_output_size = lstm_output_size
        self.output_size = output_size
        self.label_smoothing = label_smoothing

        self.batch_norm_input = nn.BatchNorm2d(1)
        self.lflbs = nn.Sequential(*[
            LocalFeatureLearningBlock2D(**config)
            for config in self.lflb_config])
        self.lstm = nn.LSTM(self.lstm_input_size, self.lstm_output_size)
        self.fc = nn.Linear(self.lstm_output_size, self.output_size)

        self.init_parameters()

    def config(self):
        config = {
            'lflb_config': self.lflb_config,
            'lstm_input_size': self.lstm_input_size,
            'lstm_output_size': self.lstm_output_size,
            'output_size': self.output_size,
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

        features = lstm_output[-1]
        features = self.fc(features)

        outputs = {
            'logits': features,
        }

        if labels is not None:
            outputs['loss'] = self.loss(features, labels)

        return outputs
