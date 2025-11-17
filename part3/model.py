
import torch
import torch.nn as nn

class SimpleRNNForTokenClassification(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim=128, hidden_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.RNN(emb_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.rnn(x)
        logits = self.fc(out)
        return logits
