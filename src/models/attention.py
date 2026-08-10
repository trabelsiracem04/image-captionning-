import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    """Additive (Bahdanau) attention over encoder spatial features.

    Inputs:
        encoder_out     [B, N, encoder_dim]   49 image-region features
        decoder_hidden  [B, decoder_dim]       LSTM hidden state
    Outputs:
        context  [B, encoder_dim]              weighted region summary
        weights  [B, N]                        softmax scores (sum to 1)
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

        torch.nn.init.xavier_uniform_(self.encoder_att.weight)
        torch.nn.init.xavier_uniform_(self.decoder_att.weight)
        torch.nn.init.xavier_uniform_(self.full_att.weight)
        torch.nn.init.zeros_(self.encoder_att.bias)
        torch.nn.init.zeros_(self.decoder_att.bias)
        torch.nn.init.zeros_(self.full_att.bias)

    def forward(self, encoder_out, decoder_hidden):
        query = self.decoder_att(decoder_hidden).unsqueeze(1)   # [B,1,attention_dim]
        projected = self.encoder_att(encoder_out)           # [B,N,attention_dim]
        scores = torch.tanh(projected + query)                # [B,N,attention_dim]
        raw = self.full_att(scores).squeeze(2)               # [B,N]
        v_t = F.softmax(raw, dim=1)                          # [B,N]
        context = (v_t.unsqueeze(2) * encoder_out).sum(dim=1)  # [B,encoder_dim]
        return context, v_t