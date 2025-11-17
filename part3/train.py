
import torch
from torch.utils.data import DataLoader
from load_data import load_conllu
from dataset import POSDataset, pad_collate
from model import SimpleRNNForTokenClassification
from collections import Counter

train_path = 'D:/Download/UD_English-EWT/UD_English-EWT/en_ewt-ud-train.conllu'
dev_path = 'D:/Download/UD_English-EWT/UD_English-EWT/en_ewt-ud-dev.conllu'

train_sents = load_conllu(train_path)
dev_sents = load_conllu(dev_path)

# build vocab
words = Counter(w for s in train_sents for w,_ in s)
tags = Counter(t for s in train_sents for _,t in s)

word_to_ix = {w:i+1 for i,(w,_) in enumerate(words.items())}
word_to_ix['<UNK>']=len(word_to_ix)+1
tag_to_ix = {t:i for i,(t,_) in enumerate(tags.items())}

train_ds = POSDataset(train_sents, word_to_ix, tag_to_ix)
dev_ds = POSDataset(dev_sents, word_to_ix, tag_to_ix)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=pad_collate)
dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=pad_collate)

model = SimpleRNNForTokenClassification(len(word_to_ix)+1, len(tag_to_ix))
opt = torch.optim.Adam(model.parameters())
crit = torch.nn.CrossEntropyLoss(ignore_index=-1)

def evaluate(loader):
    model.eval()
    correct=total=0
    with torch.no_grad():
        for x,y in loader:
            logits = model(x)
            pred = logits.argmax(-1)
            mask = y!=-1
            correct += (pred[mask]==y[mask]).sum().item()
            total += mask.sum().item()
    return correct/total

for epoch in range(2):
    model.train()
    for x,y in train_loader:
        opt.zero_grad()
        logits = model(x)
        loss = crit(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        opt.step()
    print(epoch, evaluate(train_loader), evaluate(dev_loader))

torch.save(model.state_dict(), 'D:/Download/pos_rnn/model.pt')
