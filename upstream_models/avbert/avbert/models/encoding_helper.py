import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    Ashish Vaswani et al.
    "Attention Is All You Need."
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).to(torch.float)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).to(torch.float) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x_pos = x + self.pe[:, : x.size(1)]
        return x_pos


class SummaryEncoding(nn.Module):
    """
    Implement BOS Encoding.
    """

    def __init__(
        self,
        modalities,
        d_model,
        use_mean_pooling=False,
        max_len=5000,
        layer_norm_eps=1e-12,
    ):
        super(SummaryEncoding, self).__init__()
        self.modalities = modalities
        self.bos_embeddings = nn.Embedding(len(modalities), d_model)
        self.use_mean_pooling = use_mean_pooling
        if use_mean_pooling:
            self.positional_encoding = PositionalEncoding(d_model, max_len)
            for modality in modalities:
                self.add_module(
                    f"bos_{modality}_norm", nn.LayerNorm(d_model, layer_norm_eps)
                )

    def forward(self, x, modality_idx):
        batch_size = x.size(0)
        bos = self.bos_embeddings(
            torch.full((batch_size, 1), modality_idx, dtype=torch.long, device=x.device)
        )
        if self.use_mean_pooling:
            x_pos = self.positional_encoding(x)
            x_pooled = x_pos.mean(dim=1, keepdim=True)
            bos = getattr(self, f"bos_{self.modalities[modality_idx]}_norm")(
                bos + x_pooled
            )
        x = torch.cat([bos, x], dim=1)
        return x
