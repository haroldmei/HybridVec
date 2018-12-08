from tqdm import tqdm
import sys
import torch

import numpy as np
from time import time

from torch.autograd import Variable
import torchtext.vocab as vocab

from hybridvec.loader import *
from hybridvec.config import *
from hybridvec.models import *

from web.evaluate import evaluate_on_all

import logging

#   # if not run the model on all the glove files and print the scores
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

def get_embeddings():
  config = load_config(eval=True)

  #config.batch_size = 16
  #config.dropout = 0
  #config.packing = False
  #config.input_method=INPUT_METHOD_ONE

  model_type = config.model_type

  TRAIN_FILE = 'data/glove/val_glove.%s.%sd.txt'%(config.vocab_source,config.vocab_dim)
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

  model.load_state_dict(torch.load(get_model_path(config)), strict = True)

  train_loader = get_data_loader(TRAIN_FILE,
                                 vocab_1,
                                 config.input_method,
                                 config.vocab_dim,
                                 batch_size = config.batch_size,
                                 num_workers = 0, #config.num_workers,
                                 shuffle=False,
                                 vocab_size=config.vocab_size)
  if use_gpu:
      model = model.cuda()

  model.train(False)
  out_embeddings = {}

  for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader)):
      words, inputs, lengths, labels = data
      if len(words) != config.batch_size:
        continue

      if use_gpu:
          inputs = inputs.cuda()

      outputs = model(inputs, lengths)
      for idx, word in enumerate(words):
        if model_type == 'seq2seq': #
          dec_out, dec_embd = model.get_def_embeddings(outputs) #enc_out, enc_embd, 
          #defn_en_out = torch.mean(enc_out, dim=1)[idx]
          #en_embd = torch.mean(enc_embd, dim=0)[idx]
          #out_embeddings[word] = torch.cat([defn_en_out, enc_embd[0,idx,:],enc_embd[1,idx,:]], -1)
          out_embeddings[word] = dec_embd[:,idx,:]
          #out_embeddings[word] = np.reshape(model.get_def_embeddings(outputs)[:, idx, :], (-1))
          #out_embeddings[word] = model.decoder._init_state(embd)[idx, :]
        else:
          out_embeddings[word] = model.get_def_embeddings(outputs)[idx, :]

  modelname = "{}-{}-{}".format(config.model_type, config.run_name, config.vocab_dim)
  return out_embeddings, modelname

def load_embeddings():
  a = np.load('./eval/out_embeddings.npy').item()
  return a

def main():
  embeddings, name = get_embeddings()
  results = evaluate_on_all(embeddings)
  out_fname = "{}.csv".format(name)
  logger.info("Saving results...")
  print(results)
  results.to_csv(out_fname)


"""
python ./scripts/intrinsic_evals.py --model_type seq2seq --run_name seq2seq_numworker_0 --load_epoch 20 --vocab_dim 300
python ./scripts/intrinsic_evals.py --model_type baseline --run_name baseline --load_epoch 20 --vocab_dim 300
"""
if __name__ == "__main__":
  main()
