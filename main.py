import time

import clip
import numpy as np
import PIL

import snntorch as snn
import snntorch.surrogate

import torch
import torch.nn.functional as F
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


class Autoencoder(nn.Module):

    def __init__(self, num_hidden_neurons, num_encoder_neurons, latent_dim,
                 num_steps):
        """
        Parameters
        ------------
        input_dim: the dimension of the input data
        num_hidden_neuron: number of neurons in the hidden layer within the spike encoder
        num_encoder_neuron: number of neurons in the output layer of the spike encoder
        latent_dim: dimension of the output layer of the spike decoder
        num_steps: number of time steps for the encoding process
        """
        super().__init__()
        self.cnn = efficientnet_b0()
        self.cnn.classifier = nn.Identity()
        # This only works for EfficientNet and MobileNet from TorchVision
        input_dim = 1280
        self.spike_encoder = SpikeEncoder(input_dim, num_hidden_neurons,
                                          num_encoder_neurons, beta=0.9,
                                          num_steps=num_steps)
        self.spike_output = SpikeDecoder(num_encoder_neurons, latent_dim)
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


def compute_losses(batch_images, cnn, clip_model,
                   spike_encoder, spike_decoder, recon_head, adapter, device):
    batch_images = batch_images.to(device)
    with torch.no_grad():
        cnn_features = cnn(batch_images)                                # [B, 1280]
        clip_embedding = clip_model.encode_image(batch_images).float()  # [B, 512]
    spk_rec, _ = spike_encoder(cnn_features)  # [T, B, num_encoder_neurons]
    decoded = spike_decoder(spk_rec)          # [B, latent_dim]
    recon = recon_head(decoded)               # [B, 1280]
    clip_hat = adapter(decoded)               # [B, 512]
    recon_loss = F.mse_loss(recon, cnn_features)
    semantic_loss = 1 - F.cosine_similarity(clip_hat, clip_embedding.detach()).mean()
    sparsity_loss = spk_rec.mean()
    return recon_loss, semantic_loss, sparsity_loss


def training_loop(cnn, spike_encoder, spike_decoder, recon_head, adapter,
                  clip_model, train_loader, test_loader, optimizer, num_epochs,
                  num_steps, device):
    start_time = time.time()
    trainable = [spike_encoder, spike_decoder, recon_head, adapter]
    counter = 0
    for epoch in range(num_epochs):
        for batch_images, _ in train_loader:
            recon_loss, semantic_loss, sparsity_loss = compute_losses(
                batch_images, cnn, clip_model,
                spike_encoder, spike_decoder, recon_head, adapter, device)
            total_loss = recon_loss + 0.5 * semantic_loss + 0.01 * sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if counter % 100 == 0:
                print(f"Iter {counter} | "
                      f"recon {recon_loss.item():.4f} | "
                      f"semantic {semantic_loss.item():.4f} | "
                      f"sparsity {sparsity_loss.item():.4f} | "
                      f"total {total_loss.item():.4f} | "
                      f"elapsed {round(time.time() - start_time, 2)}")
            counter += 1

        for m in trainable:
            m.eval()

        total_recon = total_semantic = total_sparsity = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch_images, _ in test_loader:
                rl, sl, spl = compute_losses(
                    batch_images, cnn, clip_model,
                    spike_encoder, spike_decoder, recon_head, adapter, device)
                total_recon += rl.item()
                total_semantic += sl.item()
                total_sparsity += spl.item()
                n_batches += 1

        avg_recon = total_recon / n_batches
        avg_semantic = total_semantic / n_batches
        avg_sparsity = total_sparsity / n_batches
        avg_total = avg_recon + 0.5 * avg_semantic + 0.01 * avg_sparsity
        print(f"Epoch {epoch} evaluation | "
              f"recon {avg_recon:.4f} | "
              f"semantic {avg_semantic:.4f} | "
              f"sparsity {avg_sparsity:.4f} | "
              f"total {avg_total:.4f}")

        for m in trainable:
            m.train()


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
    num_epochs = 30
    cnn_feature_dim = 1280
    clip_embed_dim = 512

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.requires_grad_(False)

    cnn = efficientnet_b0().to(device)
    cnn.classifier = nn.Identity()
    for p in cnn.parameters():
        p.requires_grad_(False)

    spike_encoder = SpikeEncoder(cnn_feature_dim, num_hidden_neurons,
                                 num_encoder_neurons, beta=0.9,
                                 num_steps=num_steps).to(device)
    spike_decoder = SpikeDecoder(num_encoder_neurons, latent_dim).to(device)
    recon_head = nn.Linear(latent_dim, cnn_feature_dim).to(device)
    adapter = nn.Linear(latent_dim, clip_embed_dim).to(device)

    optimizer = torch.optim.Adam(
        list(spike_encoder.parameters()) +
        list(spike_decoder.parameters()) +
        list(recon_head.parameters()) +
        list(adapter.parameters()),
        lr=5e-4, betas=(0.9, 0.999)
    )

    train_dataset = CIFAR10('cifar10', train=True, transform=preprocess,
                            download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, num_workers=4, shuffle=True)

    test_dataset = CIFAR10('cifar10', train=False, transform=preprocess,
                           download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             drop_last=True, num_workers=4, shuffle=False)

    training_loop(cnn, spike_encoder, spike_decoder, recon_head, adapter,
                  clip_model, train_loader, test_loader, optimizer, num_epochs,
                  num_steps, device)
    print("Done")