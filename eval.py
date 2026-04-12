import clip

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.models.efficientnet import efficientnet_b0, EfficientNet_B0_Weights

from main import SpikeEncoder, SpikeDecoder, CNN_TRANSFORM


if __name__ == "__main__":
    for module, dev in {torch.cuda: "cuda:0", torch.mps: "mps"}.items():
        if module.is_available():
            device = torch.device(dev)
            break
    else:
        device = torch.device("cpu")

    checkpoint = torch.load("model.pt", map_location=device)
    hp = checkpoint["hyperparameters"]

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
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
    adapter = nn.Linear(hp["latent_dim"], 512).to(device)

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
                                    dim=-1)  # [10, 512]

    test_dataset = CIFAR10("cifar10", train=False, transform=CNN_TRANSFORM,
                           download=True)
    test_loader = DataLoader(test_dataset, batch_size=128, num_workers=4,
                             shuffle=False)

    correct = 0
    total = 0
    for i, (batch_images, labels) in enumerate(test_loader):
        batch_images = batch_images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            cnn_feats = cnn(batch_images)          # [B, 1280]
            spk_rec, _ = spike_encoder(cnn_feats)  # [T, B, n_neurons]
            decoded = spike_decoder(spk_rec)       # [B, latent_dim]
            adapter_out = F.normalize(adapter(decoded), dim=-1)  # [B, 512]
            similarity = adapter_out @ text_features.T  # [B, 10]
            preds = similarity.argmax(dim=-1)           # [B]
        # break

        batch_correct = (preds == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)
        print(f"Batch {i + 1} / {len(test_loader)} | "
              f"batch acc: {batch_correct / labels.size(0) * 100:.1f}% | "
              f"running acc: {correct / total * 100:.2f}%")

    print(f"\nFinal top-1 accuracy: {correct / total * 100:.2f}% ({correct} / {total})")
