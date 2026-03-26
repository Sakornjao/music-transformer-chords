import torch
from models.transformer import ChordTransformer

class ChordPredictor:
    def __init__(self):
        self.model = self._load_model()
        self.chord_labels = [
            "C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
            "Cm","C#m","Dm","D#m","Em","Fm","F#m","Gm","G#m","Am","A#m","Bm"
        ]

    def _load_model(self):
        model = ChordTransformer()
        model.eval()
        return model

    def predict(self, chroma):
        x = torch.tensor(chroma).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(x)

        preds = torch.argmax(output, dim=-1)[0].numpy()

        chords = [self.chord_labels[i] for i in preds]

        return chords

    def predict_with_indices(self, chroma):
        x = torch.tensor(chroma).unsqueeze(0).float()

        with torch.no_grad():
            output = self.model(x)

        preds = torch.argmax(output, dim=-1)[0].numpy()

        return preds