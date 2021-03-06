import torch
import torch.nn as nn
from torch.autograd import Variable

class Def2VecModel(nn.Module):
  """ Standard sequence-to-sequence architecture with configurable encoder
    and decoder.
  """
  def __init__(self,
               vocab,
               output_size=100,
               hidden_size=150,
               embed_size=100,
               num_layers=2,
               dropout=0.0,
               use_bidirection=True,
               use_attention=True,
               cell_type='GRU',
               use_cuda=True,
               use_packing=False,
               max_length=784):
    super(Def2VecModel, self).__init__()
    self.use_packing = use_packing
    self.use_cuda = use_cuda
    self.vocab_size = len(vocab.stoi)
    self.embeddings = nn.Embedding(self.vocab_size + 1, embed_size, padding_idx=0)
    self.embeddings.weight.data[1:,:].copy_(vocab.vectors)
    self.embeddings.weight.data[0:,:] = 0
    self.embed_size = embed_size
    self.num_layers = num_layers
    self.output_size = output_size
    self.hidden_size = hidden_size
    self.use_attention = use_attention
    self.use_bidirection = use_bidirection
    self.cell_type = cell_type
    if cell_type == 'GRU':
        self.cell = nn.GRU(embed_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    elif cell_type == 'LSTM':
        self.cell = nn.LSTM(embed_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    elif cell_type == 'RNN':
        self.cell = nn.RNN(embed_size,
                           hidden_size,
                           num_layers,
                           batch_first=True,
                           dropout=dropout,
                           bidirectional=use_bidirection)
    else:
        self.baseline = nn.Linear(embed_size, hidden_size)
    if use_attention:
        self.attn = nn.Linear((2 if use_bidirection else 1) * hidden_size, 1)
        self.attn_softmax = nn.Softmax(dim=1)
    self.output_layer = nn.Linear((2 if use_bidirection else 1) * hidden_size, output_size)

  def forward(self, inputs, lengths = None, return_attn = False):
    inputs = Variable(inputs)
    batch_size, input_size = inputs.shape
    embed = self.embeddings(inputs.view(-1, input_size)).view(batch_size, input_size, -1)
    if self.use_packing:
      embed = nn.utils.rnn.pack_padded_sequence(embed, lengths, batch_first=True)
    h0 = Variable(torch.zeros(self.num_layers * (2 if self.use_bidirection else 1),
                              batch_size, self.hidden_size))
    if self.use_cuda:
      h0 = h0.cuda()
    if self.cell_type:
        cell_outputs, _ = self.cell(embed, h0)
    else:
        cell_outputs = self.baseline(embed)
    if self.use_packing:
      cell_outputs, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(
                                        cell_outputs, batch_first=True)
    if self.use_attention:
        logits = self.attn(cell_outputs)
        softmax = self.attn_softmax(logits)
        mean = torch.sum(softmax * cell_outputs, dim=1)
    else:
        mean = torch.mean(cell_outputs, dim=1)
    our_embedding = self.output_layer(mean)
    if return_attn:
        return our_embedding, softmax
    else:
        return our_embedding
