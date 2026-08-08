import torch.nn as nn
import torchvision.models as tv

SUPPORTED_ENCODERS = ("resnet50",)


class ResNetEncoder(nn.Module):
    def __init__(self, encoder="resnet50", pretrained=True):
        super().__init__()
        if encoder not in SUPPORTED_ENCODERS:
            raise ValueError(
                f"unsupported encoder {encoder!r}; supported: {list(SUPPORTED_ENCODERS)}"
            )
        weights = tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.cnn = tv.resnet50(weights=weights)
        self.backbone = nn.Sequential(
            self.cnn.conv1,
            self.cnn.bn1,
            self.cnn.relu,
            self.cnn.maxpool,
            self.cnn.layer1,
            self.cnn.layer2,
            self.cnn.layer3,
            self.cnn.layer4,
        )
        self.cnn = None

        self.feature_dim = 2048
        self.grid_h = 7
        self.grid_w = 7
        self.num_spatial = self.grid_h * self.grid_w

        if not pretrained:
            self.freeze()
        else:
            self.fine_tune = False

    def forward(self, x):
        return self.backbone(x)

    def spatial_features(self, x):
        features = self.backbone(x)
        b, c, h, w = features.shape
        return features.permute(0, 2, 3, 1).reshape(b, h * w, c)

    def global_features(self, x):
        features = self.backbone(x)
        return features.mean(dim=(2, 3))

    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.fine_tune = False

    def unfreeze_layer4(self):
        for name, param in self.backbone.named_parameters():
            param.requires_grad = name.startswith("4.")
        self.fine_tune = True


def build_encoder(cfg) -> ResNetEncoder:
    return ResNetEncoder(
        encoder=cfg.model.encoder,
        pretrained=cfg.model.encoder_pretrained,
    )