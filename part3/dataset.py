
import torch
from torch.utils.data import Dataset

class POSDataset(Dataset):
    def __init__(self, sentences, word_to_ix, tag_to_ix):
        self.sentences = sentences
        self.word_to_ix = word_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words, tags = zip(*self.sentences[idx])
        w_idx = [self.word_to_ix.get(w, self.word_to_ix['<UNK>']) for w in words]
        t_idx = [self.tag_to_ix[t] for t in tags]
        return torch.tensor(w_idx), torch.tensor(t_idx)

def pad_collate(batch):
    words, tags = zip(*batch)
    words_p = torch.nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=0)
    tags_p = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-1)
    return words_p, tags_p
