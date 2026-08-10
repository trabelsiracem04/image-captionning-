import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.attention import Attention


class Decoder(nn.Module):
    """Attention-based LSTM decoder with beta-gated context.

    Inputs for training (teacher forcing):
        encoder_out       [B, N, encoder_dim]   ResNet spatial features
        encoded_captions  [B, T]                token ids (<SOS>, words, <EOS>)
    Outputs:
        logits   [B, T, vocab_size]
        weights  [B, T, N]

    Inference uses init_hidden_state + step.
    """

    def __init__(self, encoder_dim, embed_dim, hidden_dim, attention_dim, vocab_size, dropout):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.attention_dim = attention_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(
            encoder_dim=encoder_dim,
            decoder_dim=hidden_dim,
            attention_dim=attention_dim,
        )

        self.embedding_bn = nn.BatchNorm1d(embed_dim)
        self.context_bn = nn.BatchNorm1d(encoder_dim)
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTMCell(input_size=embed_dim + encoder_dim, hidden_size=hidden_dim)
        self.f_beta = nn.Linear(hidden_dim, encoder_dim, bias=False)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)

    def init_hidden_state(self, encoder_out):
        mean_features = encoder_out.mean(dim=1)
        return self.init_h(mean_features), self.init_c(mean_features)

    def step(self, encoder_out, word_embedding, h, c):
        context, weights = self.attention(encoder_out, h)
        beta = torch.sigmoid(self.f_beta(h))
        context = beta * context
        lstm_input = torch.cat(
            [self.embedding_bn(word_embedding), self.context_bn(context)], dim=1
        )
        h, c = self.lstm(self.dropout(lstm_input), (h, c))
        logits = self.fc(self.dropout(h))
        return logits, h, c, weights

    def forward(self, encoder_output, encoded_captions):
        batch_size, length = encoded_captions.shape

        embeddings = self.embedding(encoded_captions)
        h, c = self.init_hidden_state(encoder_output)

        logits = []
        weights = []
        for t in range(length):
            logits_t, h, c, weights_t = self.step(
                encoder_output, embeddings[:, t], h, c
            )
            logits.append(logits_t)
            weights.append(weights_t)

        return torch.stack(logits, dim=1), torch.stack(weights, dim=1)


def build_decoder(cfg, vocab) -> Decoder:
    return Decoder(
        encoder_dim=2048,
        embed_dim=cfg.model.embed_dim,
        hidden_dim=cfg.model.hidden_dim,
        attention_dim=cfg.model.attention_dim,
        vocab_size=len(vocab),
        dropout=cfg.model.dropout,
    )