import numpy as np

import snntorch as snn
import snntorch.surrogate

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b0
from torchvision.datasets import CIFAR10


class SpikeEncoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, beta, num_steps):
        super().__init__()
        self.num_steps = num_steps
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, input):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        mem2_rec = []

        for step in range(self.num_steps):
            cur1 = self.fc1(input)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)


class SpikeDecoder(nn.Module):

    def __init__(self, num_neurons, latent_dim):
        super().__init__()
        # Project from the hidden space to the latent space linearly
        self.mlp = nn.Sequential(nn.Linear(num_neurons, latent_dim),
                                 nn.LayerNorm(latent_dim))

    def forward(self, input):
        firing_rates = input.mean(dim=0)  # aggregate over time
        return self.mlp(firing_rates)


class NN(nn.Module):

    def __init__(self, num_hidden_neuron, num_encoder_neuron, latent_dim,
                 num_steps):
        """
        input_dim: the dimension of the input data
        num_hidden_neuron: the hidden dimension
        """
        super().__init__()
        self.cnn = efficientnet_b0()
        self.cnn.classifier = nn.Identity()
        # This only works for EfficientNet and MobileNet from TorchVision
        input_dim = 1280
        self.spike_encoder = SpikeEncoder(input_dim, num_hidden_neuron,
                                          num_encoder_neuron, beta=0.9,
                                          num_steps=num_steps)
        self.spike_output = SpikeDecoder(num_encoder_neuron, latent_dim)
        # Adapter to direct output from the spike train output to CLIP
        # self.adapter = ...
        self.composite = nn.Sequential(self.cnn,
                                       self.spike_encoder,
                                       self.spike_output)

    def forward(self, input):
        cnn_out = self.cnn(input)
        spk_rec, mem_rec = self.spike_encoder(cnn_out)
        _ = self.spike_output(spk_rec)
        return spk_rec, mem_rec


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


def print_batch_accuracy(net, data, targets, batch_size, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def training_loop(net, train_loader, test_loader, loss,
                  optimizer, batch_size, num_epochs):
    dtype = torch.float
    loss_hist = []
    test_loss_hist = []
    counter = 0
    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data)

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data)

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    print(f"Epoch {epoch}, Iteration {iter_counter}")
                    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
                    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
                    print_batch_accuracy(net, data, targets, batch_size,
                                         train=True)
                    print_batch_accuracy(net, test_data, test_targets,
                                         batch_size, train=False)
                    print("\n")
                counter += 1
                iter_counter +=1


if __name__ == "__main__":
    for module, dev in {torch.cuda: 'cuda:0', torch.mps: 'mps'}.items():
        if module.is_available():
            device = torch.device(dev)
            break
    else:
        device = torch.device('cpu')

    num_encoder_neurons = 16
    num_hidden_neurons = 100
    latent_dim = 16
    num_steps = 20
    batch_size = 128
    num_epochs = 10

    raw_dataset = CIFAR10('cifar10', train=True, download=True,
                          transform=transforms.ToTensor())
    net = NN(num_hidden_neurons, num_encoder_neurons, latent_dim,
             num_steps).to(device)

    normalization = compute_normalization(raw_dataset)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*normalization)])
    # transform = transforms.Compose([transforms.Resize((224, 224)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0, 0, 0), (1, 1, 1))])


    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4,
                                 betas=(0.9, 0.999))

    train_dataset = CIFAR10('cifar10', train=True, transform=transform,
                            download=True)
    test_dataset = CIFAR10('cifar10', train=False, transform=transform,
                           download=True)

    train_dataset_loader = DataLoader(train_dataset, drop_last=True,
                                      batch_size=batch_size)
    test_dataset_loader = DataLoader(test_dataset, drop_last=True,
                                      batch_size=batch_size)
    training_loop(net, train_dataset_loader, test_dataset_loader, loss,
                  optimizer, batch_size, num_epochs)