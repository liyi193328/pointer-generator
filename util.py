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
import tensorflow as tf
import time
import os
import six
import numpy as np
FLAGS = tf.app.flags.FLAGS

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
  punt_list = ',.!?:;~，。！？：；～' # string 必须要解码为 unicode 才能进行匹配
  if six.PY2:
    punt_list = punt_list.decode("utf-8")
  for word in words:
    # if six.PY2 and type(word) == str:
    word = word.encode("utf-8")
    if word in punt_list:
      sents.append(words[start:i + 1])
      start = i + 1  # start标记到下一句的开头
      i += 1
    else:
      i += 1  # 若不是标点符号，则字符位置继续前移
  if start < len(words):
    sents.append(words[start:])  # 这是为了处理文本末尾没有标点符号的情况
  return sents

if __name__ == "__main__":
  print( cut_sentence(["我爱","中国","。","我爱","中国"]) )