import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import constraints
from pyro.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform
from models.transforms import FoldedDistribution_New, RectifiedNormal, CensoredNormal


class BNNmlp(PyroModule):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()

        self.activation = nn.ReLU()  # could also be ReLU or LeakyReLU
        assert in_dim > 0 and out_dim > 0 and hid_dim > 0 and n_hid_layers > 0  # make sure the dimensions are valid

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]

        # layer_list = [HiddenLayer(self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
        #               range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(1., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y=None):
        dim = x.shape[0]
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # infer the response noise
        with pyro.plate("data", dim):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu


class BNNUNet(PyroModule):
    def __init__(self, in_channels=6, out_channels=1, features=[16, 32, 64], prior_scale=5.):
        super().__init__()
        self.down_convs = PyroModule[nn.ModuleList]()
        self.up_convs = PyroModule[nn.ModuleList]()
        self.down_batch = PyroModule[nn.ModuleList]()
        self.up_batch = PyroModule[nn.ModuleList]()
        self.convs_inup = PyroModule[nn.ModuleList]()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.7)  # or another dropout rate
        # self.batchnorm = nn.BatchNorm1d(feature)

        # Construct the down-sampling path
        for feature in features:
            conv = PyroModule[nn.Conv1d](in_channels, feature, kernel_size=3, padding=1)
            conv.weight = PyroSample(dist.Normal(0., prior_scale).expand([feature, in_channels, 3]).to_event(3))
            conv.bias = PyroSample(dist.Normal(0., prior_scale).expand([feature]).to_event(1))
            self.down_convs.append(conv)
            batchnorm = PyroModule[nn.BatchNorm1d](num_features=feature)
            batchnorm.weight = PyroSample(
                prior=dist.Normal(0., prior_scale).expand([feature]).to_event(1))
            batchnorm.bias = PyroSample(
                prior=dist.Normal(0., prior_scale).expand([feature]).to_event(1))
            self.down_batch.append(batchnorm)
            # conv = PyroModule[nn.Conv1d](feature, feature, kernel_size=3, padding=1)
            # conv.weight = PyroSample(dist.Normal(0., prior_scale).expand([feature, in_channels, 3]).to_event(3))
            # conv.bias = PyroSample(dist.Normal(0., prior_scale).expand([feature]).to_event(1))
            # self.down_convs.append(conv)
            in_channels = feature

        # Construct the up-sampling path
        reversed_features = list(reversed(features))
        for idx in range(len(reversed_features) - 1):
            input_channels = reversed_features[idx]
            output_channels = reversed_features[idx]//2
            if idx < len(reversed_features) - 2:
                upconv = PyroModule[nn.ConvTranspose1d](input_channels, output_channels, kernel_size=3, padding = 1, stride=2, output_padding=1)
            else:
                upconv = PyroModule[nn.ConvTranspose1d](input_channels, output_channels, kernel_size=3, padding = 0, stride=2, output_padding=0)
            upconv.weight = PyroSample(dist.Normal(0., prior_scale).expand([input_channels, output_channels, 3]).to_event(3))
            upconv.bias = PyroSample(dist.Normal(0., prior_scale).expand([output_channels]).to_event(1))
            self.up_convs.append(upconv)
            conv = PyroModule[nn.Conv1d](output_channels*2, output_channels, kernel_size=3, padding="same")
            conv.weight = PyroSample(dist.Normal(0., prior_scale).expand([output_channels, input_channels, 3]).to_event(3))
            conv.bias = PyroSample(dist.Normal(0., prior_scale).expand([output_channels]).to_event(1))
            self.convs_inup.append(conv)
            batchnorm = PyroModule[nn.BatchNorm1d](num_features=output_channels)
            batchnorm.weight = PyroSample(
                prior=dist.Normal(0., prior_scale).expand([output_channels]).to_event(1))
            batchnorm.bias = PyroSample(
                prior=dist.Normal(0., prior_scale).expand([output_channels]).to_event(1))
            self.up_batch.append(batchnorm)


        # Final output layer
        self.final_conv1 = PyroModule[nn.Conv1d](features[0], output_channels, kernel_size=3)
        self.final_conv1.weight = PyroSample(dist.Normal(0., prior_scale).expand([output_channels, features[0], 1]).to_event(3))
        self.final_conv1.bias = PyroSample(dist.Normal(0., prior_scale).expand([output_channels]).to_event(1))
        self.final_conv2 = PyroModule[nn.Conv1d](features[0], out_channels, kernel_size=1)
        self.final_conv2.weight = PyroSample(dist.Normal(0., prior_scale).expand([out_channels, features[0], 1]).to_event(3))
        self.final_conv2.bias = PyroSample(dist.Normal(0., prior_scale).expand([out_channels]).to_event(1))
        
    def forward(self, x, y=None):
        # Down-sampling path
        skip_connections = []
        for idx, (conv, batchnorm) in enumerate(zip(self.down_convs, self.down_batch)):
            if idx < len(self.down_convs) - 1:
                x = self.activation(batchnorm(conv(x)))
                if idx < len(self.down_convs) - 1:
                    skip_connections.append(x)
                x = self.dropout(x) 
                x = self.pool(x)
            else:
                x = self.activation(batchnorm(conv(x)))
                x = self.dropout(x) 

        # Up-sampling path
        for idx, (upconv, conv, batchnorm) in enumerate(zip(self.up_convs, self.convs_inup, self.up_batch)):
            x = upconv(x)
            skip_connection = skip_connections[-(idx+1)]
            x = torch.cat((x, skip_connection), dim=1)
            x = self.activation(batchnorm(conv(x)))
            x = self.dropout(x) 

        mu = self.final_conv2(self.activation(batchnorm(self.final_conv1(x)))).squeeze()
        sigma = pyro.sample("sigma", dist.Gamma(0.5, 1))  # Infer the response noise
        sigma = sigma.expand(mu.shape)
        # extended_sigmoid = ComposeTransform([SigmoidTransform(), AffineTransform(loc=0, scale=2)])
        # truncated_normal = dist.FoldedDistribution(dist.Normal(mu, sigma), low=0)
        with pyro.plate("data", size=x.size(0), dim=-2):
            # base_dist = dist.Normal(mu, sigma)
            obs = pyro.sample("obs", CensoredNormal(loc=mu, scale=sigma), obs=y)
            # obs = pyro.sample("obs", FoldedDistribution_New(dist.Normal(mu, sigma)), obs=y)
            # obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu


class BNN(PyroModule):
    def __init__(self, in_dim=7, out_dim=1, hid_dim=10, n_hid_layers=5, prior_scale=5.):
        super().__init__()

        self.activation = nn.Tanh()  # could also be ReLU or LeakyReLU
        assert in_dim > 0 and out_dim > 0 and hid_dim > 0 and n_hid_layers > 0  # make sure the dimensions are valid

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx]) for idx in
                      range(1, len(self.layer_sizes))]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(dist.Normal(0., prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx])).expand(
                [self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]]).to_event(2))
            layer.bias = PyroSample(dist.Normal(0., prior_scale).expand([self.layer_sizes[layer_idx + 1]]).to_event(1))

    def forward(self, x, y=None):
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x).squeeze()  # hidden --> output
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1))  # infer the response noise

        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma), obs=y)
        return mu