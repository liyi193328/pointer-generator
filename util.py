#encoding=utf-8

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

"""This file contains some utility functions"""
import time
import os
import six
import codecs
import glob
import numpy as np
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

def get_dir_or_file_path(dir_or_path, max_deep=1):
  if os.path.exists(dir_or_path) == False:
    raise ValueError("{} not exists".format(dir_or_path))
  if os.path.isdir(dir_or_path):
    all_paths = [os.path.join(dir_or_path, name) for name in os.listdir(dir_or_path)]
  else:
    all_paths = glob.glob(dir_or_path)
  return all_paths

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config

def load_ckpt(saver, sess):
  """Load checkpoint from the train directory and restore it to saver and sess, waiting 10 secs in the case of failure. Also returns checkpoint name."""
  while True:
    try:
      train_dir = os.path.join(FLAGS.log_root, "train")
      ckpt_state = tf.train.get_checkpoint_state(train_dir)
      tf.logging.info('Loading checkpoint %s', ckpt_state.model_checkpoint_path)
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      return ckpt_state.model_checkpoint_path
    except:
      tf.logging.info("Failed to load checkpoint from %s. Sleeping for %i secs...", train_dir, 10)
      time.sleep(10)

def cut_sentence(words):

  start = 0
  i = 0  # 记录每个字符的位置
  sents = []
  punt_list = u'.!?;~。！？～' # string 必须要解码为 unicode 才能进行匹配
  for word in words:
    if six.PY2 and type(word) == str:
      word = word.decode("utf-8")
    # print(type(word))
    if word in punt_list:
      sents.append(words[start:i + 1])
      start = i + 1  # start标记到下一句的开头
      i += 1
    else:
      i += 1  # 若不是标点符号，则字符位置继续前移
  if start < len(words):
    sents.append(words[start:])  # 这是为了处理文本末尾没有标点符号的情况
  return sents

def read_sum_sents(file_path,sent_token=False):
  f = codecs.open(file_path, "r", "utf-8")
  sum_sents = []
  while True:
    line = f.readline()
    if line == "":
      break
    sents = line.strip()
    if sent_token:
      sents = sents.split(" ")
    sum_sents.append(sents)
  return sum_sents

if __name__ == "__main__":
  print( cut_sentence(["我爱","中国","。","我爱","中国"]) )
