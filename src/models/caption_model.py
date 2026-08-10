import torch.nn as nn

from src.models.decoder import build_decoder
from src.models.encoder import build_encoder


class CaptionModel(nn.Module):
    """End-to-end image captioning model: CNN encoder + attention decoder.

    forward is the teacher-forcing path used for training.
    """

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, encoded_captions):
        features = self.encoder.spatial_features(images)
        logits, weights = self.decoder(features, encoded_captions)
        return logits, weights, features

    def encode(self, images):
        return self.encoder.spatial_features(images)

    def step(self, features, word_embedding, h, c):
        return self.decoder.step(features, word_embedding, h, c)

    def init_hidden_state(self, features):
        return self.decoder.init_hidden_state(features)


def build_caption_model(cfg, vocab) -> CaptionModel:
    encoder = build_encoder(cfg)
    decoder = build_decoder(cfg, vocab)
    if cfg.training.fine_tune:
        encoder.unfreeze_layer4()
    else:
        encoder.freeze()
    return CaptionModel(encoder, decoder)