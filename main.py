from numpy.typing import NDArray
import snntorch

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.datasets import CIFAR10


class SpikeEncoder(nn.Module):

    def __init__(self, input_dim, num_neurons, beta, num_steps):
        super().__init__()
        # Number of time-steps
        self.num_steps = num_steps
        # Models synaptic inputs. Which and how the input is actually
        # mapped/directed to which neuron is determined by the data
        self.fc = nn.Linear(input_dim, num_neurons)
        # Models the leaky integrate-and-fire model
        self.lif = snntorch.Leaky(beta=beta)

    def forward(self, input):
        mem = self.lif.init_leaky()
        spikes = []
        h = self.fc(input)
        for _ in range(self.num_steps):
            spike, mem = self.lif(h, mem)
            spikes.append(spike)
        return torch.stack(spikes, dim=-1)  # [batch, num_neurons, num_steps]


class SpikeDecoder(nn.Module):

    def __init__(self, num_neurons, latent_dim):
        super().__init__()
        # Project from the hidden space to the latent space linearly
        self.mlp = nn.Sequential(nn.Linear(num_neurons, latent_dim),
                                 nn.LayerNorm(latent_dim))

    def forward(self, input):
        firing_rates = input.mean(dim=-1)  # aggregate over time
        return self.mlp(firing_rates)


class NN(nn.Module):

    def __init__(self, num_neurons, latent_dim, num_steps):
        """
        input_dim: the dimension of the input data
        hidden_dim: the hidden dimension
        """
        super().__init__()
        self.cnn = efficientnet_b0()
        self.cnn.classifier = nn.Identity()
        # This only works for EfficientNet and MobileNet from TorchVision
        input_dim = 1280
        self.spike_encoder = SpikeEncoder(input_dim, num_neurons, beta=0.9,
                                          num_steps=num_steps)
        self.spike_output = SpikeDecoder(num_neurons, latent_dim)
        # Adapter to direct output from the spike train output to CLIP
        # self.adapter = ...
        self.composite = nn.Sequential(self.cnn, self.spike_encoder, self.spike_output)

    def forward(self, input):
        return self.composite(input)


def compute_normalization(dataset):
    # May need to set batch size
    n = 0
    means = torch.zeros(3)
    var = torch.zeros(3)
    for images, _ in DataLoader(dataset, batch_size=256):
        n += images.shape[0] * images.shape[2] * images.shape[3]
        means += torch.sum(images, dim=(0, 2, 3))
    means /= n

    for images, _ in DataLoader(dataset, batch_size=256):
        for i in range(3):
            var[i] += torch.sum((images[:, i, :, :] - means[i]) ** 2)
    var /= n
    return means, var ** 0.5


if __name__ == "__main__":
    for module, dev in {torch.cuda: 'cuda:0', torch.mps: 'mps'}.items():
        if module.is_available():
            device = torch.device(dev)
            break
    else:
        device = torch.device('cpu')

    num_neurons = 256
    latent_dim = 16
    num_steps = 20
    training_batch_size = 128

    raw_dataset = CIFAR10('cifar10', train=True, transform=transforms.ToTensor(),
                          download=True)
    net = NN(num_neurons, latent_dim, num_steps).to(device)
    normalization = compute_normalization(raw_dataset)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(*normalization)])
    dataset0 = CIFAR10('cifar10', train=True, transform=transform, download=True)
    dataset1 = CIFAR10('cifar10', train=False, transform=transform, download=True)
    for input, output in DataLoader(dataset0, batch_size=training_batch_size,
                                    drop_last=True):
        # Call model on input to get output
        output_ = net(input.to(device))