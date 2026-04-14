import clip

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.linear_model import LogisticRegression

from train import SpikeEncoder, SpikeDecoder, CNN_TRANSFORM, CNN_TRAIN_TRANSFORM, DualTransformDataset


def extract_embeddings(loader, cnn, clip_model, spike_encoder, spike_decoder, adapter, device):
    all_clip, all_adapter, all_latent, all_labels = [], [], [], []
    for cnn_imgs, clip_imgs, labels in loader:
        cnn_imgs = cnn_imgs.to(device)
        clip_imgs = clip_imgs.to(device)
        with torch.no_grad():
            clip_embs = clip_model.encode_image(clip_imgs).float()  # [B, clip_embed_dim]
            cnn_feats = cnn(cnn_imgs)                               # [B, 1280]
            spk_rec, _ = spike_encoder(cnn_feats)                   # [T, B, n_neurons]
            latent = spike_decoder(spk_rec)                         # [B, latent_dim]
            adapter_out = adapter(latent)                           # [B, clip_embed_dim]
        all_clip.append(clip_embs.cpu())
        all_adapter.append(adapter_out.cpu())
        all_latent.append(latent.cpu())
        all_labels.append(labels)
    return (torch.cat(all_clip).numpy(),
            torch.cat(all_adapter).numpy(),
            torch.cat(all_latent).numpy(),
            torch.cat(all_labels).numpy())


def linear_probe(train_embs, train_labels, test_embs, test_labels):
    probe = LogisticRegression(max_iter=1000, C=1.0)
    probe.fit(train_embs, train_labels)
    return probe.score(test_embs, test_labels)


if __name__ == "__main__":
    for module, dev in {torch.cuda: "cuda:0", torch.mps: "mps"}.items():
        if module.is_available():
            device = torch.device(dev)
            break
    else:
        device = torch.device("cpu")

    checkpoint = torch.load("model.pt", map_location=device)
    hp = checkpoint["hyperparameters"]

    clip_model, preprocess = clip.load(hp.get("clip_model", "ViT-B/32"), device=device)
    clip_model.requires_grad_(False)

    cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).to(device)
    cnn.classifier = nn.Identity()
    cnn.requires_grad_(False)

    spike_encoder = SpikeEncoder(input_dim=1280,
                                 hidden_dim=hp["n_hidden"],
                                 output_dim=hp["n_neurons"],
                                 beta=0.9,
                                 num_steps=hp["num_steps"]).to(device)
    n3 = checkpoint["spike_decoder"]["mlp.0.weight"].shape[0]
    spike_decoder = SpikeDecoder(n2=hp["n_neurons"],
                                 n1=hp["n_hidden"],
                                 n4=hp["latent_dim"],
                                 n3=n3).to(device)
    clip_embed_dim = hp.get("clip_embed_dim", 512)
    adapter = nn.Linear(hp["latent_dim"], clip_embed_dim).to(device)

    spike_encoder.load_state_dict(checkpoint["spike_encoder"])
    spike_decoder.load_state_dict(checkpoint["spike_decoder"])
    adapter.load_state_dict(checkpoint["adapter"])

    for m in [cnn, clip_model, spike_encoder, spike_decoder, adapter]:
        m.eval()
        m.requires_grad_(False)

    train_dataset = DualTransformDataset(
        CIFAR10("cifar10", train=True, transform=None, download=True),
        cnn_transform=CNN_TRAIN_TRANSFORM,
        clip_transform=preprocess,
    )
    test_dataset = DualTransformDataset(
        CIFAR10("cifar10", train=False, transform=None, download=True),
        cnn_transform=CNN_TRANSFORM,
        clip_transform=preprocess,
    )
    train_loader = DataLoader(train_dataset, batch_size=128, num_workers=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4, shuffle=False)

    print(f"Extracting train embeddings ({len(train_dataset)} samples)...")
    tr_clip, tr_adapter, tr_latent, tr_labels = extract_embeddings(
        train_loader, cnn, clip_model, spike_encoder, spike_decoder, adapter, device)

    print(f"Extracting test embeddings ({len(test_dataset)} samples)...")
    te_clip, te_adapter, te_latent, te_labels = extract_embeddings(
        test_loader, cnn, clip_model, spike_encoder, spike_decoder, adapter, device)

    print("Fitting probes...")
    acc_clip = linear_probe(tr_clip, tr_labels, te_clip, te_labels)
    acc_adapter = linear_probe(tr_adapter, tr_labels, te_adapter, te_labels)
    acc_latent = linear_probe(tr_latent, tr_labels, te_latent, te_labels)

    print("\nLinear Probe Results (CIFAR-10, 10 classes)")
    print("--------------------------------------------")
    print(f"clip         ({tr_clip.shape[1]:4d}-dim): {acc_clip * 100:.2f}%  <- ceiling")
    print(f"adapter_out  ({tr_adapter.shape[1]:4d}-dim): {acc_adapter * 100:.2f}%  <- CLIP-projected spike")
    print(f"latent       ({tr_latent.shape[1]:4d}-dim): {acc_latent * 100:.2f}%  <- bottleneck")
    print("--------------------------------------------")
    print(f"spike bottleneck cost vs CLIP: {(acc_clip - acc_adapter) * 100:.2f}%")
    print(f"adapter gain over latent:      {(acc_adapter - acc_latent) * 100:.2f}%")
