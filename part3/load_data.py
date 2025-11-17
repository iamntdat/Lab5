
import re

def load_conllu(path):
    sentences=[]
    sent=[]
    with open(path, encoding='utf8') as f:
        for line in f:
            line=line.strip()
            if not line:
                if sent: sentences.append(sent); sent=[]
                continue
            if line.startswith('#'): continue
            parts=line.split('\t')
            if len(parts)>3:
                word=parts[1]
                tag=parts[3]
                sent.append((word, tag))
    if sent: sentences.append(sent)
    return sentences
