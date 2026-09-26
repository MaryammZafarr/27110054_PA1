import torch
import torch.nn as nn
import torchvision.models as tvm

from data.transforms import normalize_imagenet, normalize_clip


class ResNet50Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2
        net = tvm.resnet50(weights=weights)
        self.stem = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        self.pool = net.avgpool
        self.out_dim = 2048
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = normalize_imagenet(x)
        feat = self.stem(x)
        feat = self.pool(feat).flatten(1)
        return feat


class ViTB16Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1
        self.net = tvm.vit_b_16(weights=weights)
        self.out_dim = self.net.hidden_dim
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = normalize_imagenet(x)
        n = self.net
        x = n._process_input(x)
        cls = n.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = n.encoder(x)
        return x[:, 0]   


class OpenCLIPExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai")
        self.model = model
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.clip_preprocess = preprocess
        self.out_dim = model.visual.output_dim
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def encode_image(self, x):
        x = normalize_clip(x)
        feat = self.model.encode_image(x)
        return feat / feat.norm(dim=-1, keepdim=True)

    forward = encode_image

    @torch.no_grad()
    def zero_shot_logits(self, x, class_names, template="a photo of a {}.", device="cuda"):
        img_feat = self.encode_image(x)
        prompts = [template.format(c) for c in class_names]
        tokens = self.tokenizer(prompts).to(device)
        text_feat = self.model.encode_text(tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * img_feat @ text_feat.t()


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, feat):
        return self.fc(feat)


def build_backbones(device: str = "cuda") -> dict:
    resnet = ResNet50Extractor().to(device)
    vit = ViTB16Extractor().to(device)
    clip = OpenCLIPExtractor().to(device)
    return {"resnet50": resnet, "vit_b16": vit, "clip_vitb32": clip}