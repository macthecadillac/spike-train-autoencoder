import clip

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import CIFAR10
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights

from main import SpikeEncoder, SpikeDecoder, CNN_TRANSFORM, DualTransformDataset


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
    adapter_hidden = checkpoint["adapter"]["0.weight"].shape[0]
    clip_embed_dim = hp.get("clip_embed_dim", 512)
    adapter = nn.Sequential(nn.Linear(hp["latent_dim"], adapter_hidden),
                            nn.ReLU(),
                            nn.Linear(adapter_hidden, clip_embed_dim)).to(device)

    spike_encoder.load_state_dict(checkpoint["spike_encoder"])
    spike_decoder.load_state_dict(checkpoint["spike_decoder"])
    adapter.load_state_dict(checkpoint["adapter"])

    for m in [cnn, spike_encoder, spike_decoder, adapter]:
        m.eval()
        m.requires_grad_(False)

    CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                       "dog", "frog", "horse", "ship", "truck"]

    prompts = clip.tokenize([f"a photo of a {c}" for c in CIFAR10_CLASSES]).to(device)
    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(prompts).float(),
                                    dim=-1)  # [10, clip_embed_dim]

    test_dataset = DualTransformDataset(
        CIFAR10("cifar10", train=False, transform=None, download=True),
        cnn_transform=CNN_TRANSFORM,
        clip_transform=preprocess,
    )
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4,
                             shuffle=False)

    clip_correct = 0
    spike_correct = 0
    agree = 0
    total = 0
    for i, (cnn_imgs, clip_imgs, labels) in enumerate(test_loader):
        cnn_imgs = cnn_imgs.to(device)
        clip_imgs = clip_imgs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # Direct CLIP path
            clip_img_feats = F.normalize(clip_model.encode_image(clip_imgs).float(), dim=-1)
            clip_preds = (clip_img_feats @ text_features.T).argmax(dim=-1)

            # Spike model path
            cnn_feats = cnn(cnn_imgs)                          # [B, 1280]
            spk_rec, _ = spike_encoder(cnn_feats)              # [T, B, n_neurons]
            decoded = spike_decoder(spk_rec)                   # [B, latent_dim]
            adapter_out = F.normalize(adapter(decoded), dim=-1)
            spike_preds = (adapter_out @ text_features.T).argmax(dim=-1)

        batch_clip_correct = (clip_preds == labels).sum().item()
        batch_spike_correct = (spike_preds == labels).sum().item()
        batch_agree = (clip_preds == spike_preds).sum().item()
        batch_size = labels.size(0)

        clip_correct += batch_clip_correct
        spike_correct += batch_spike_correct
        agree += batch_agree
        total += batch_size

        print(f"Batch {i + 1} / {len(test_loader)} | "
              f"clip: {batch_clip_correct / batch_size * 100:.1f}% | "
              f"spike: {batch_spike_correct / batch_size * 100:.1f}% | "
              f"agree: {batch_agree / batch_size * 100:.1f}%")

    print(f"\nFinal results ({total} images):")
    print(f"  Direct CLIP accuracy:  {clip_correct / total * 100:.2f}% ({clip_correct} / {total})")
    print(f"  Spike model accuracy:  {spike_correct / total * 100:.2f}% ({spike_correct} / {total})")
    print(f"  Agreement rate:        {agree / total * 100:.2f}% ({agree} / {total})")
