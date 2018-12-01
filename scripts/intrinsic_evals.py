from __future__ import print_function
from tqdm import tqdm
import sys
import torch.nn.functional as F
import collections
import traceback
import torch
import torch.optim as optim
import torch.nn as nn

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import numpy as np
from time import time

from torch.autograd import Variable
import torchtext.vocab as vocab
from tensorboardX import SummaryWriter
from pytorch_monitor import monitor_module, init_experiment

from hybridvec.eval.eval_scripts import evaluate_on_all

import json
import argparse
from collections import OrderedDict
from hybridvec.loader import *
from hybridvec.config import eval_config
from hybridvec.models import Seq2seq, BaselineModel, EncoderRNN, DecoderRNN

import logging

#   # if not run the model on all the glove files and print the scores
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)


# # runs over all the words in glove and returns embeddings for each one
# def get_embeddings():
#   #check if there is a local file first


#   # if not run the model on all the glove files and print the scores

def get_args():
    """
    Gets the run_name, run_comment, and epoch of the model being evaluated
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_name")
    parser.add_argument("run_comment")
    parser.add_argument("epoch")
    parser.add_argument("--verbose", default=True)
    args = parser.parse_args()
    return (args.run_name, args.run_comment, args.epoch, args.verbose)

def load_config():
    """
    Load in the right config file from desired model to evaluate
    """
    run_name, run_comment, epoch, verbose = get_args()
    name = run_name + '-' + run_comment

    path = "outputs/def2vec/logs/{}/config.json".format(name)
    config = None
    with open(path) as f:
        config = dict(json.load(f))
        config = eval_config(config, run_name, run_comment, epoch, verbose)
    return (config, name, run_name)

def get_embeddings():
  config, name, model_type = load_config()
  TRAIN_FILE = 'data/glove/train_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
  vocab_1 = vocab.GloVe(name=config.vocab_source, dim=config.vocab_dim)
  use_gpu = torch.cuda.is_available()
  print("Using GPU:", use_gpu)

  if model_type == 'baseline': 
      model = BaselineModel(vocab_1, 
                           config = config, 
                           use_cuda = use_gpu)

  elif model_type == 'seq2seq':
      encoder = EncoderRNN(config = config,
                           variable_lengths = False, 
                           embedding = None)
      decoder = DecoderRNN(config = config)
      model = Seq2seq(encoder = encoder, 
                           decoder=decoder)



  model.load_state_dict(torch.load(config.save_path), strict = True)

  train_loader = get_data_loader(TRAIN_FILE,
                                 vocab_1,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = config.num_workers,
                                 shuffle=False,
                                 vocab_size=config.vocab_size)
  if use_gpu:
      model = model.cuda()
  model.train(False)

  running_loss = 0.0
  n_batches = 0
  out_embeddings = {}
  pred_defns = {}
  out_defns = {}

  for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
      words, inputs, lengths, labels = data
      labels = Variable(labels)

      if use_gpu:
          inputs = inputs.cuda()
          labels = labels.cuda()

      outputs = model(inputs, lengths)
      for idx, word in enumerate(words):
        out_embeddings[word] = model.get_def_embeddings(outputs)[idx, :]

  # out_dir = "outputs/def2vec/checkpoints/{}".format(name)
  #       if not os.path.exists(out_dir):
  #           os.makedirs(out_dir)
  # np.save("./outputs/def2vec/checkpoints/{}/output_embeddings.npy".format(name), out_embeddings)
  return out_embeddings


def load_embeddings():
  a = np.load('./eval/out_embeddings.npy').item()
  return a

def main():
  embeddings = get_embeddings()
  #embeddings = load_embeddings()
  results = evaluate_on_all(embeddings)
  out_fname = "results.csv"
  logger.info("Saving results...")
  print(results)
  results.to_csv(out_fname)




if __name__ == "__main__":
  main()
