"""
FCN: fullly convolutional network from ["Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline"]
code is from (https://github.com/okrasolar/pytorch-timeseries).
"""
import torch
from torch import nn
import torch.nn.functional as F


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """
    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        return self.layers(x)


class FCNBaseline(nn.Module):
    """A PyTorch implementation of the FCN Baseline
    From https://arxiv.org/abs/1909.04939

    Attributes
    ----------
    sequence_length:
        The size of the input sequence
    num_pred_classes:
        The number of output classes
    """

    def __init__(self, in_channels: int, num_pred_classes: int = 1) -> None:
        super().__init__()

        # for easier saving and loading
        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels, 128, 8, 1),
            ConvBlock(128, 256, 5, 1),
            ConvBlock(256, 128, 3, 1),
        ])
        self.final = nn.Linear(128, num_pred_classes)

    def forward(self, x: torch.Tensor, return_feature_vector: bool = False) -> torch.Tensor:  # type: ignore
        out = self.layers(x)

        if return_feature_vector:
            return out.mean(dim=-1)
        else:
            return self.final(out.mean(dim=-1))


if __name__ == '__main__':
    # toy input
    B, C, L = 32, 2, 256
    x = torch.rand((B, C, L))

    # model
    fcn = FCNBaseline(in_channels=C, num_pred_classes=10)

    # forward
    yhat = fcn(x)
    print('1) yhat.shape:', yhat.shape)

    yhat = fcn(x, return_feature_vector=True)
    print('2) yhat.shape:', yhat.shape)
