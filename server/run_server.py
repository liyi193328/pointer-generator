# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This is the top-level file to train, evaluate or test your summarization model"""
from __future__ import absolute_import
from __future__ import print_function

import sys
import os
from server.summarizer import Summarizer

sys.path.append('..')
import tensorflow as tf
from collections import namedtuple
from data import Vocab
from model import SummarizationModel
from decode import BeamSearchDecoder
from flask import Flask
import optparse

import run_summarization

FLAGS = tf.app.flags.FLAGS
from flask import render_template, request

with open('server/templates/fish_article.txt') as f:
  default_article = f.read()

"""
python -m server.run_server --vocab_path=/home/bigdata/active_project/run_tasks/query_rewrite/stable/stable_single/copy_data_format/vocab/shared.vob.txt \
--log_root=/home/bigdata/active_project/run_tasks/query_rewrite/stable/stable_single/copy_model/ --exp_name=vocab_3w_256 --vocab_size=30000 --hidden_dim=256 \
--max_enc_steps=20 --max_dec_steps=30 --batch_size=1 --single_pass=True --mode=decode --restore_best_model --Serving --coverage
"""

def setup_summarizer():
  tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
  tf.logging.info('Starting seq2seq_attention ')

  FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  vocab = Vocab(FLAGS.vocab_path, FLAGS.vocab_size)  # create a vocabulary

  # If in decode mode, set batch_size = beam_size
  # Reason: in decode mode, we decode one example at a time.
  # On each step, we have beam_size-many hypotheses in the beam, so we need to make a batch of these hypotheses.
  FLAGS.batch_size = FLAGS.beam_size

  # Make a namedtuple hps, containing the values of the hyperparameters that the model needs
  hparam_list = ['mode', 'lr', 'adagrad_init_acc', 'rand_unif_init_mag', 'trunc_norm_init_std', 'max_grad_norm',
                 'hidden_dim', 'emb_dim', 'batch_size', 'max_dec_steps', 'max_enc_steps', 'coverage', 'cov_loss_wt',
                 'pointer_gen']
  hps_dict = {}
  for key, val in FLAGS.__flags.items():  # for each flag
    if key in hparam_list:  # if it's in the list
      hps_dict[key] = val  # add it to the dict
  hps = namedtuple("HParams", list(hps_dict.keys()))(**hps_dict)

  tf.set_random_seed(111)  # a seed value for randomness

  if hps.mode != 'decode':
    raise ValueError("The 'mode' flag must be decode for serving")
  decode_model_hps = hps  # This will be the hyperparameters for the decoder model
  decode_model_hps = hps._replace(
    max_dec_steps=1)  # The model is configured with max_dec_steps=1 because we only ever run one step of the decoder at a time (to do beam search). Note that the batcher is initialized with max_dec_steps equal to e.g. 100 because the batches need to contain the full summaries
  serving_device = '/cpu:0'
  model = SummarizationModel(decode_model_hps, vocab, default_device=serving_device)
  decoder = BeamSearchDecoder(model, None, vocab)
  return Summarizer(decoder, vocab=vocab, hps=hps)

summarizer = setup_summarizer()

app = Flask(__name__)


@app.route('/')
def index():
  return render_template('index.html', summary='N/A', article=default_article)


@app.route('/', methods=['POST'])
def index_post():
  article = request.form['article']
  summarized_text_list = summarizer.summarize(article)
  summarized_text = "\n".join(summarized_text_list)
  return render_template('index.html', summary=summarized_text, article=article)


@app.route('/summarize/<text>')
def summarize(text):
  return summarizer.summarize(text)


app.run(host='0.0.0.0')
