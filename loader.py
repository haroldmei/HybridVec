import numpy as np
import string
from definitions import get_wordnet_definition
import linecache
import torch
from torch.utils.data import Dataset, DataLoader


class DefinitionsDataset(Dataset):

  def __init__(self, vocab_file, vocab):
    self.vocab_file = vocab_file
    self.vocab_len = len(vocab.stoi)
    self.vocab = vocab
    self.idx_offset = 0

  def __len__(self):
    return self.vocab_len

  def __getitem__(self, idx):
    """
    Return (definition, embedding)
    """
    word,embedding = self.get_vocab_pair(idx + self.idx_offset)
    definition = None
    while definition is None:
      self.idx_offset += 1
      word,embedding = self.get_vocab_pair(idx + self.idx_offset)
      definition = get_wordnet_definition(word)
      if definition is None:
          continue
      try:
        definition = definition[list(definition.keys())[0]][0]
        exclude = set(string.punctuation)
        definition = [self.vocab.stoi["".join([c for c in word.lower() if \
                                               c not in exclude])] for \
                      word in definition.split()]
      except Exception as e:
        print('Error in lookup')
        print(e)
        definition = None
    return (np.array(definition), embedding.astype(np.float32))

  def get_vocab_pair(self, idx):
    word = None
    while word is None:
        line = linecache.getline(self.vocab_file, idx + self.idx_offset + 1)
        splitLine = line.split()
        if len(line) == 0:
            self.idx_offset += 1
            continue
        try:
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
        except Exception as e:
            print(e)
            print(splitLine)
            self.idx_offset += 1
    return (word, embedding)


def get_data_loader(vocab_file, vocab):
  dataset = DefinitionsDataset(vocab_file, vocab)
  return DataLoader(dataset, batch_size=1, num_workers=1, shuffle=True)