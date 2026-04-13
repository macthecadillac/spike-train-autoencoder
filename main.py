import time

import clip
import numpy as np

import snntorch as snn
import snntorch.surrogate

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset


CNN_TRANSFORM = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

CNN_TRAIN_TRANSFORM = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])


class DualTransformDataset(Dataset):
    """Wraps a dataset (with no transform) and applies separate transforms
    for the CNN and CLIP model."""

    def __init__(self, base_dataset, cnn_transform, clip_transform):
        self.base = base_dataset
        self.cnn_transform = cnn_transform
        self.clip_transform = clip_transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.cnn_transform(img), self.clip_transform(img), label


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


class LeakyIntegrator(nn.Module):

    def __init__(self, beta=0.95):
        super().__init__()
        self.register_buffer("beta", torch.as_tensor(beta, dtype=torch.float))

    def forward(self, inputs):  # inputs: [T, B, N]
        mem = torch.zeros_like(inputs[0])
        mem_seq = []
        for inp in inputs:
            mem = self.beta.clamp(0, 1) * mem + inp
            mem_seq.append(mem)
        return torch.stack(mem_seq, dim=0)  # [T, B, N]


class SpikeDecoder(nn.Module):

    def __init__(self, n1, n2, n4, beta_lif=0.9, beta_li=0.95, n3=64):
        super().__init__()
        self.fc_lif = nn.Linear(n2, n1)
        self.lif = snn.Leaky(beta=beta_lif)
        self.fc_li = nn.Linear(n1, n2)
        self.li = LeakyIntegrator(beta=beta_li)
        self.mlp = nn.Sequential(nn.Linear(n2, n3),
                                 nn.ReLU(),
                                 nn.Linear(n3, n4),
                                 nn.LayerNorm(n4))

    def forward(self, spk_rec):  # spk_rec: [T, B, N2]
        num_steps = spk_rec.shape[0]
        mem = self.lif.init_leaky()
        lif_spk_seq = []
        for t in range(num_steps):
            cur = self.fc_lif(spk_rec[t])  # [B, N2]
            spk, mem = self.lif(cur, mem)  # [B, N2]
            lif_spk_seq.append(spk)
        lif_spk_seq = torch.stack(lif_spk_seq, dim=0)   # [T, B, N2]
        li_input = self.fc_li(lif_spk_seq)              # [T, B, N1]
        li_mem = self.li(li_input)                      # [T, B, N1]
        return self.mlp(li_mem.mean(dim=0))              # [B, latent_dim]


def print_batch_accuracy(net, data, targets, batch_size, train=False):
    output, _ = net(data)
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def compute_losses(cnn_images, clip_images, cnn, clip_model,
                   spike_encoder, spike_decoder, recon_head, adapter, device):
    cnn_images = cnn_images.to(device)
    clip_images = clip_images.to(device)
    with torch.no_grad():
        cnn_features = cnn(cnn_images)                                   # [B, 1280]
        clip_embedding = clip_model.encode_image(clip_images).float()    # [B, 512]
    spk_rec, _ = spike_encoder(cnn_features)  # [T, B, num_encoder_neurons]
    decoded = spike_decoder(spk_rec)          # [B, latent_dim]
    recon = recon_head(decoded)               # [B, 1280]
    clip_hat = adapter(decoded)               # [B, 512]
    recon_loss = F.mse_loss(recon, cnn_features)
    semantic_loss = 1 - F.cosine_similarity(clip_hat, clip_embedding.detach()).mean()
    sparsity_loss = spk_rec.mean()
    return recon_loss, semantic_loss, sparsity_loss


def training_loop(cnn, spike_encoder, spike_decoder, recon_head, adapter,
                  clip_model, train_loader, test_loader, optimizer, scheduler,
                  num_epochs, num_steps, device):
    start_time = time.time()
    trainable = [spike_encoder, spike_decoder, recon_head, adapter]
    counter = 0
    recon_history, semantic_history, sparsity_history = [], [], []
    for epoch in range(num_epochs):
        for cnn_imgs, clip_imgs, _ in train_loader:
            recon_loss, semantic_loss, sparsity_loss = compute_losses(
                cnn_imgs, clip_imgs, cnn, clip_model,
                spike_encoder, spike_decoder, recon_head, adapter, device)
            total_loss = recon_loss + semantic_loss + 0.05 * sparsity_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if counter % 20 == 0:
                recon_history.append(recon_loss.item())
                semantic_history.append(semantic_loss.item())
                sparsity_history.append(sparsity_loss.item())
            counter += 1

        for m in trainable:
            m.eval()

        total_recon = total_semantic = total_sparsity = 0.0
        n_batches = 0
        with torch.no_grad():
            for cnn_imgs, clip_imgs, _ in test_loader:
                rl, sl, spl = compute_losses(
                    cnn_imgs, clip_imgs, cnn, clip_model,
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
              f"total {avg_total:.4f} | "
              f"elapsed {round(time.time() - start_time, 2)}")

        scheduler.step()

        for m in trainable:
            m.train()

    return recon_history, semantic_history, sparsity_history


if __name__ == "__main__":
    for module, dev in {torch.cuda: 'cuda:0', torch.mps: 'mps'}.items():
        if module.is_available():
            device = torch.device(dev)
            break
    else:
        device = torch.device('cpu')

    num_encoder_neurons = 256
    num_hidden_neurons = 512
    latent_dim = 128
    num_steps = 20
    batch_size = 128
    num_epochs = 100
    cnn_feature_dim = 1280
    clip_embed_dim = 512
    decoder_beta_lif = 0.9
    decoder_beta_li = 0.95
    decoder_mlp_hidden = 128

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.requires_grad_(False)

    cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
    cnn.classifier = nn.Identity()
    for p in cnn.parameters():
        p.requires_grad_(False)

    spike_encoder = SpikeEncoder(cnn_feature_dim, num_hidden_neurons,
                                 num_encoder_neurons, beta=0.9,
                                 num_steps=num_steps).to(device)
    spike_decoder = SpikeDecoder(n2=num_encoder_neurons,
                                 n1=num_hidden_neurons,
                                 n4=latent_dim,
                                 beta_lif=decoder_beta_lif,
                                 beta_li=decoder_beta_li,
                                 n3=decoder_mlp_hidden).to(device)
    recon_head = nn.Linear(latent_dim, cnn_feature_dim).to(device)
    adapter = nn.Sequential(nn.Linear(latent_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, clip_embed_dim)).to(device)

    optimizer = torch.optim.Adam(list(spike_encoder.parameters()) +
                                 list(spike_decoder.parameters()) +
                                 list(recon_head.parameters()) +
                                 list(adapter.parameters()),
                                 lr=5e-4, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-5)

    train_dataset = DualTransformDataset(
        CIFAR10('cifar10', train=True, transform=None, download=True),
        cnn_transform=CNN_TRAIN_TRANSFORM, clip_transform=preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              drop_last=True, num_workers=4, shuffle=True)

    test_dataset = DualTransformDataset(
        CIFAR10('cifar10', train=False, transform=None, download=True),
        cnn_transform=CNN_TRANSFORM, clip_transform=preprocess)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             drop_last=True, num_workers=4, shuffle=False)

    recon_h, semantic_h, sparsity_h = training_loop(
        cnn, spike_encoder, spike_decoder, recon_head, adapter,
        clip_model, train_loader, test_loader, optimizer, scheduler,
        num_epochs, num_steps, device)

    np.savez("loss_history.npz",
             recon=recon_h,
             semantic=semantic_h,
             sparsity=sparsity_h)

    model = {
        "spike_encoder": spike_encoder.state_dict(),
        "spike_decoder": spike_decoder.state_dict(),
        "recon_head": recon_head.state_dict(),
        "adapter": adapter.state_dict(),
        "optimizer": optimizer.state_dict(),
        "hyperparameters": {
            "n_neurons": num_encoder_neurons,
            "n_hidden": num_hidden_neurons,
            "latent_dim": latent_dim,
            "num_steps": num_steps,
            "num_epochs": num_epochs,
            "decoder_mlp_hidden": decoder_mlp_hidden,
        },
    }
    torch.save(model, "model.pt")

    print("Done")