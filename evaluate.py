from __future__ import print_function
from tqdm import tqdm
import sys
import torch.nn.functional as F
import collections
import traceback
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, accuracy_score, recall_score, mean_squared_error
from time import time
from model import Def2VecModel
from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment
from loader import *
from config import eval_config

DEBUG_LOG = False
config = eval_config()


TEST_FILE = 'data/glove/test_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)


def get_word(word):
    return vocab.vectors[vocab.stoi[word]]

def closest(vec, n=10):
    """
    Find the closest words for a given vector
    """
    all_dists = [(w, torch.dist(vec, get_word(w))) for w in vocab.itos]
    return sorted(all_dists, key=lambda t: t[1])[:n]

if __name__ == "__main__":

    vocab = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
    use_gpu = torch.cuda.is_available()
    print("Using GPU:", use_gpu)

    model = Def2VecModel(vocab,
                         embed_size = config.vocab_dim,
                         output_size = config.vocab_dim,
                         hidden_size = config.hidden_size,
                         use_packing = config.packing,
                         use_bidirection = config.use_bidirection,
                         use_attention = config.use_attention,
                         cell_type = config.cell_type,
                         use_cuda = use_gpu)
    model.load_state_dict(torch.load(config.save_path))
    test_loader = get_data_loader(TEST_FILE,
                                   vocab,
                                   config.__method,
                                   config.vocab_dim,
                                   batch_size = config.batch_size,
                                   num_workers = config.num_workers,
                                   shuffle=False)

    if use_gpu:
        model = model.cuda()
    criterion = nn.MSELoss()
    model.train(False)

    running_loss = 0.0
    n_batches = 0
    out_embeddings = {}
    out_attns = {}
    out_defns = {}

    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader)):
        words, inputs, lengths, labels = data
        labels = Variable(labels)

        if use_gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs, attns = model(inputs, return_attn=True)
        loss = criterion(outputs, labels)

        running_loss += loss.data[0]
        n_batches += 1

        for word, embed, attn, inp in zip(words,
                                          outputs.data.cpu(),
                                          attns.data.cpu().squeeze(2),
                                          inputs.cpu()):
            out_embeddings[word] = embed.numpy()
            out_attns[word] = attn.numpy()
            out_defns[word] = " ".join([vocab.itos[i - 1] for i in inp])

    print("L2 loss:", running_loss / n_batches)
    np.save("eval/out_embeddings.npy", out_embeddings)
    np.save("eval/out_attns.npy", out_attns)
    np.save("eval/out_defns.npy", out_defns)
