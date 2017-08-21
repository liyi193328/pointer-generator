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

"""This file contains code to run beam search decoding, including running ROUGE evaluation and producing JSON datafiles for the in-browser attention visualizer, which can be found here https://github.com/abisee/attn_vis"""

import os
import time
import tensorflow as tf
import beam_search
import data
import json
import rouge
import rouge_score
import util
import logging
import codecs
import six
import shutil
import numpy as np
from collections import OrderedDict
try:
  from pyltp import SentenceSplitter
except ImportError:
  SentenceSplitter = None

FLAGS = tf.app.flags.FLAGS

SECS_UNTIL_NEW_CKPT = 60  # max number of seconds before loading new checkpoint

def copy_model_post_fn(text, source_text=None):
  """unique sentence end !!!! => !
  :param line:
  :return:
  """
  tokens = text.strip().split(" ")
  source_tokens = source_text.split(" ")
  source_token_cnt = {}
  for token in source_tokens:
    if token not in source_token_cnt:
      source_token_cnt[token] = 0
    source_token_cnt[token] += 1
  commons = ["!", "?"]

  if len(tokens) == 0:
    return ""
  else:
    last_token = tokens[-1]
    new_last_token = []
    char_set = set()
    for char in last_token:
      if char not in new_last_token:
        new_last_token.append(char)
    new_last_token = "".join(new_last_token)
    tokens[-1] = new_last_token

    new_tokens = []
    for i, token in enumerate(tokens):
      if i > 0 and tokens[i] == tokens[i-1]:
        if tokens[i] in commons:
          continue
        if tokens[i] not in source_token_cnt or source_token_cnt[tokens[i]] < 2:
          continue
      new_tokens.append(token)
  return " ".join(new_tokens)

class BeamSearchDecoder(object):
  """Beam search decoder."""

  def __init__(self, model, batcher, vocab):
    """Initialize decoder.

    Args:
      model: a Seq2SeqAttentionModel object.
      batcher: a Batcher object.
      vocab: Vocabulary object
    """
    self._model = model
    self._model.build_graph()
    self._batcher = batcher
    self._vocab = vocab
    self._saver = tf.train.Saver() # we use this to load checkpoints for decoding
    self._sess = tf.Session(config=util.get_config())

    # Load an initial checkpoint to use for decoding
    ckpt_path = util.load_ckpt(self._saver, self._sess)

    if batcher is None:
      return

    if FLAGS.single_pass and FLAGS.Serving == False:
      # Make a descriptive decode directory name
      ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
      if FLAGS.infer_dir is None:
        self._decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
      else:
        self._decode_dir = os.path.join(FLAGS.infer_dir, get_decode_dir_name(ckpt_name))
      if os.path.exists(self._decode_dir):
        if FLAGS.clear_decode_dir is True:
          shutil.rmtree(self._decode_dir)
        else:
          raise Exception("single_pass decode directory %s should not already exist" % self._decode_dir)

    else: # Generic decode dir name
      self._decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not FLAGS.Serving:
      if not os.path.exists(self._decode_dir): os.makedirs(self._decode_dir)

    if FLAGS.single_pass and FLAGS.Serving == False:
      # Make the dirs to contain output written in the correct format for pyrouge
      self._rouge_ref_path = os.path.join(self._decode_dir, "ref.txt")
      self._rouge_dec_path = os.path.join(self._decode_dir, "infer.txt")
      self._article_with_gen_abs_path = os.path.join(self._decode_dir, "article_ref_infer.txt")
      self._ref_f = codecs.open(self._rouge_ref_path, "w", "utf-8")
      self._dec_f = codecs.open(self._rouge_dec_path, "w", "utf-8")
      self._gen_f = codecs.open(self._article_with_gen_abs_path, "w","utf-8")

      self._rouge_result_path = os.path.join(self._decode_dir, "rouge_result.log")

  def decode(self, input_batch = None, beam_nums_to_return=1, *args,**kwargs):
    """Decode examples until data is exhausted (if FLAGS.single_pass) and return, or decode indefinitely, loading latest checkpoint at regular intervals"""
    t0 = time.time()
    counter = 0
    while True:
      if input_batch is not None:
        batch = input_batch
      else:
        batch = self._batcher.next_batch()  # 1 example repeated across batch
      if batch is None: # finished decoding dataset in single_pass mode
        assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
        tf.logging.info("Decoder has finished reading dataset for single_pass.")
        tf.logging.info("Output has been saved in %s,%s and %s. Now starting ROUGE eval...", self._rouge_ref_path, self._rouge_dec_path, self._article_with_gen_abs_path)
        results_dict = rouge_eval_write(self._rouge_ref_path, self._rouge_dec_path, self._rouge_result_path)
        return

      if FLAGS.single_pass:
        if FLAGS.max_infer_batch is not None and counter >= FLAGS.max_infer_batch:
          tf.logging.info("up to max_infer_batch={}, begin to eval rogue".format(FLAGS.max_infer_batch))
          results_dict = rouge_eval_write(self._rouge_ref_path, self._rouge_dec_path, self._rouge_result_path)
          return

      original_article = batch.original_articles[0]  # string
      original_abstract = batch.original_abstracts[0]  # string
      original_abstract_sents = batch.original_abstracts_sents[0]  # list of strings

      article_withunks = data.show_art_oovs(original_article, self._vocab) # string
      abstract_withunks = data.show_abs_oovs(original_abstract, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None)) # string

      # Run beam search to get {beam_nums_to_return} best Hypothesis
      best_hyp_list = beam_search.run_beam_search(self._sess, self._model, self._vocab, batch, beam_nums_to_return=beam_nums_to_return)

      decoded_output_list = []

      for bx, best_hyp in enumerate(best_hyp_list):
        # Extract the output ids from the hypothesis and convert back to words
        output_ids = [int(t) for t in best_hyp.tokens[1:]]
        decoded_words = data.outputids2words(output_ids, self._vocab, (batch.art_oovs[0] if FLAGS.pointer_gen else None))

        # Remove the [STOP] token from decoded_words, if necessary
        try:
          fst_stop_idx = decoded_words.index(data.STOP_DECODING) # index of the (first) [STOP] symbol
          decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
          decoded_words = decoded_words
        decoded_output = ' '.join(decoded_words) # single string
        decoded_output = copy_model_post_fn(decoded_output, source_text=original_article)
        decoded_output_list.append(decoded_output)

      if input_batch is not None:  # Finish decoding given single example
        print_results(article_withunks, abstract_withunks, decoded_output_list) # log output to screen
        return decoded_output_list

      for decoded_output, best_hyp in zip(decoded_output_list, best_hyp_list):
        if FLAGS.single_pass:
          self.write_for_rouge(original_abstract_sents, decoded_words, counter, article=original_article) # write ref summary and decoded summary to file, to eval with pyrouge later
          counter += 1 # this is how many examples we've decoded
        else:
          print_results(article_withunks, abstract_withunks, decoded_output) # log output to screen
          self.write_for_attnvis(article_withunks, abstract_withunks, decoded_words, best_hyp.attn_dists, best_hyp.p_gens) # write info to .json file for visualization tool

          # Check if SECS_UNTIL_NEW_CKPT has elapsed; if so return so we can load a new checkpoint
          t1 = time.time()
          if t1-t0 > SECS_UNTIL_NEW_CKPT:
            tf.logging.info('We\'ve been decoding with same checkpoint for %i seconds. Time to load new checkpoint', t1-t0)
            _ = util.load_ckpt(self._saver, self._sess)
            t0 = time.time()

  def write_for_rouge(self, reference_sents, decoded_words, ex_index, article=None):
    """Write output to file in correct format for eval with pyrouge. This is called in single_pass mode.

    Args:
      reference_sents: list of strings
      decoded_words: list of strings
      ex_index: int, the index with which to label the files
    """

    # First, divide decoded output into sentences
    decoded_sents = []

    if SentenceSplitter is None:
      decoded_sents = util.cut_sentence(decoded_words)
      for i in range(len(decoded_sents)):
        decoded_sents[i] = " ".join(decoded_sents[i])
    else:
      decoded_text = " ".join(decoded_words)
      decoded_sents = SentenceSplitter.split(decoded_text.encode("utf-8"))

    # pyrouge calls a perl script that puts the data into HTML files.
    # Therefore we need to make our output HTML safe.
    decoded_sents = [make_html_safe(w) for w in decoded_sents]
    reference_sents = [make_html_safe(w) for w in reference_sents]

    # Write to file

    if article is not None:
      with codecs.open(self._article_with_gen_abs_path, "a", "utf-8") as f:
        f.write("article:\n")
        f.write(article + "\n")
        f.write("ref:\n")
        for idx, sent in enumerate(reference_sents):
          f.write(sent + "\n")
        f.write("gen:\n")
        for idx, sent in enumerate(decoded_sents):
          if six.PY2 and type(sent) == str:
            sent = sent.decode("utf-8")
          f.write(sent + "\n")
        f.write("\n")

    with codecs.open(self._rouge_ref_path, "a", "utf-8") as f:
      for idx,sent in enumerate(reference_sents):
        if six.PY2 and type(sent) == str:
          reference_sents[idx] = sent.decode("utf-8")
      reference_sents_str = "".join(reference_sents)
      f.write(reference_sents_str + "\n")

    with codecs.open(self._rouge_dec_path, "a", "utf-8") as f:
      for idx,sent in enumerate(decoded_sents):
        if six.PY2 and type(sent) == str:
          decoded_sents[idx] = sent.decode("utf-8")
      decoded_sents_str = "".join(decoded_sents)
      f.write(decoded_sents_str + "\n")

    tf.logging.info("Wrote example %i to file" % ex_index)

  def write_for_attnvis(self, article, abstract, decoded_words, attn_dists, p_gens):
    """Write some data to json file, which can be read into the in-browser attention visualizer tool:
      https://github.com/abisee/attn_vis

    Args:
      article: The original article string.
      abstract: The human (correct) abstract string.
      attn_dists: List of arrays; the attention distributions.
      decoded_words: List of strings; the words of the generated summary.
      p_gens: List of scalars; the p_gen values. If not running in pointer-generator mode, list of None.
    """
    article_lst = article.split() # list of words
    decoded_lst = decoded_words # list of decoded words
    to_write = {
        'article_lst': [make_html_safe(t) for t in article_lst],
        'decoded_lst': [make_html_safe(t) for t in decoded_lst],
        'abstract_str': make_html_safe(abstract),
        'attn_dists': attn_dists
    }
    if FLAGS.pointer_gen:
      to_write['p_gens'] = p_gens
    output_fname = os.path.join(self._decode_dir, 'attn_vis_data.json')
    with open(output_fname, 'w') as output_file:
      json.dump(to_write, output_file, ensure_ascii=False, indent=2)
    tf.logging.info('Wrote visualization data to %s', output_fname)


def print_results(article, abstract, decoded_output):
  """Prints the article, the reference summmary and the decoded summary to screen"""
  print("")
  tf.logging.info('ARTICLE:  %s', article)
  tf.logging.info('REFERENCE SUMMARY: %s', abstract)
  tf.logging.info('GENERATED SUMMARY: %s', decoded_output)
  print("")


def make_html_safe(s):
  """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
  s.replace("<", "&lt;")
  s.replace(">", "&gt;")
  return s


def rouge_eval_write(ref_path, dec_path, result_path):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  ref_sum_sents = util.read_sum_sents(ref_path, sent_token=False)
  des_sum_sents = util.read_sum_sents(dec_path, sent_token=False)
  assert  len(ref_sum_sents) == len(des_sum_sents)
  rou = rouge.Rouge()
  scores = rou.get_scores(des_sum_sents, ref_sum_sents)
  ave_scores = rou.get_scores(des_sum_sents, ref_sum_sents, avg=True)
  rouge_eval_f = codecs.open(result_path, "w", "utf-8")
  result = OrderedDict()
  result["ave_scores"] = ave_scores
  result["detail_scores"] = scores
  json.dump(result, rouge_eval_f, indent=2)
  tf.logging.info(ave_scores)
  tf.logging.info("write eval result to {}".format(result_path))
  return result

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s_%imaxenc_%ibeam_%imindec_%imaxdec" % (dataset, FLAGS.max_enc_steps, FLAGS.beam_size, FLAGS.min_dec_steps, FLAGS.max_dec_steps)
  if ckpt_name is not None:
    dirname += "_%s" % ckpt_name
  return dirname
