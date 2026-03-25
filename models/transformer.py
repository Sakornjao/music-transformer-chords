import torch
import torch.nn as nn

class ChordTransformer(nn.Module):
    def __init__(self, input_dim=12, d_model=64, nhead=4, num_layers=2, num_classes=24):
        super().__init__()

        self.input_fc = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x