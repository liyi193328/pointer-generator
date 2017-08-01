#encoding=utf-8

import sys
import os
import six
import click
import codecs
import hashlib
import struct
import charset
import subprocess
import collections
import multiprocessing as MP
import tensorflow as tf
import numpy as np
from tensorflow.core.example import example_pb2

import pyltp
import util

LTP_DATA_DIR = os.environ.get("LTP_DATA_DIR","/home/bigdata/software/LTP/ltp_data")
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

from pyltp import Segmentor
from pyltp import SentenceSplitter
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型


# words = segmentor.segment('元芳你怎么看')  # 分词
# print ("  ".join(words))

PARA_TAG = "</para>"
VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

abs_filter_words = u"导读|导语|引导语|摘要|本文导读|小编导读|内容摘要|概述|小育导读|核心导读|内容提要|内容简介|文章导读|涂乐导读|小编导语|亿欧导读"
filter_words = abs_filter_words.split("|")

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")


def preprocess_abs_text(abs_text):
  i = 0
  for filter_word in filter_words:
    if abs_text.startswith(filter_word) == True:
      i = len(filter_word)
      if i >= len(abs_text):
        return ""
      while i < len(abs_text) and not charset.is_chinese(abs_text[i]):
        i += 1
      break
  return abs_text[i:]

def token_file(file_path, token_path_or_handle, head_index=0, abs_index=1, article_index=2, delimiter="\t"):
  save = False
  if type(token_path_or_handle) == six.text_type:
    fo = codecs.open(token_path_or_handle, "w", "utf-8")
    save = True
  else:
    fo = token_path_or_handle
  all_lines = codecs.open(file_path, "r", "utf-8").readlines()
  for i, line in enumerate(all_lines):
    t = line.strip().split(delimiter)
    new_line = []
    for j, every_element in enumerate(t):
      if j == article_index:
        every_element = every_element.replace(PARA_TAG, "")
      if j == abs_index:
        every_element = preprocess_abs_text(every_element)
      every_element = every_element.strip()
      if every_element == "":
        tokens = ""
      else:
        try:
          tokens = segmentor.segment(every_element.encode("utf-8"))
        except Exception:
          print(line)
          print(every_element)
          import traceback
          traceback.print_exc()
          raise Exception()
      tokens_str = " ".join(tokens)
      new_line.append(tokens_str)
    line_str = delimiter.join(new_line) + "\n"
    fo.write(six.text_type(line_str))
  print("{} done".format(file_path))
  if save:
    print("tokenize {}, save to {}".format(file_path, token_path_or_handle))

def token_file_or_dir(file_or_dir, token_path_or_dir, abs_index=1, article_index = 2, delimiter="\t", filters=None, pnums = MP.cpu_count() - 1):
  if os.path.isdir(file_or_dir):
    path_list = []
    for i, filename in enumerate(os.listdir(file_or_dir)):
      path = os.path.join(file_or_dir, filename)
      if filters == None or filters in filename:
        if os.path.isfile(path):
          path_list.append(path)
  else:
    path_list = [file_or_dir]
  print("will tokenize {} files: {}...".format(len(path_list), path_list[0:3]))
  if os.path.isdir(token_path_or_dir):
    pnums = min(pnums, len(path_list))
    print("will use {} pros to tokenize".format(pnums))
    pool = MP.Pool(pnums)
    pros = []
    for i, file_path in enumerate(path_list):
      token_path = os.path.join(token_path_or_dir, os.path.basename(file_path).split(".")[0] + ".token")
      pro = pool.apply_async(token_file, args=(file_path, token_path,), kwds={"article_index":article_index,"delimiter":delimiter})
      pros.append(pro)
    for i, pro in enumerate(pros):
      res = pro.get()
  else:
    save_path = token_path_or_dir
    fo = codecs.open(save_path, "w", "utf-8")
    for i, file_path in enumerate(path_list):
      token_file(file_path, fo, article_index=article_index, delimiter=delimiter)
    fo.close()

def preprocess_abs_tokens(abs, article=None, max_substring_sents=1):

  pre_abs = abs

  if pre_abs.strip() == "":
    print("abs is emtpty")
    return False

  try:
    abs_sents = SentenceSplitter.split(pre_abs.encode("utf-8"))
  except Exception:
    print(abs)
    import traceback
    traceback.print_exc()
    raise  Exception()

  new_abs_list = []
  substring_sents = 0
  for sent in abs_sents:
    if article is not None and sent in article:
      substring_sents += 1
    if substring_sents > max_substring_sents:
      return False
    sent_str = " ".join([SENTENCE_START, sent, SENTENCE_END])
    new_abs_list.append(sent_str)
    pre_abs = " ".join(new_abs_list)

  if isinstance(pre_abs, six.text_type) == False:
    pre_abs = pre_abs.decode("utf-8")
  return pre_abs

def preprocess_article_tokens(article, min_tokens=50, max_tokens=None, token_split=" "):
  if article.strip() == "":
    return False
  if isinstance(article, six.text_type) == False:
    article = article.encode("utf-8")
  tokens = article.split(token_split)
  if len(tokens) < min_tokens:
    return False
  if max_tokens is not None and len(tokens) > max_tokens:
    return False
  return article

def get_article_abs(line, delimeter="\t", abs_index=1, article_index=2):
  eles = line.strip().split(delimeter)
  try:
    or_article, or_abs = eles[article_index], eles[abs_index]
  except IndexError:
    print(line)
    print("Index error")
    print(len(eles))
    return [False, False]
  article = preprocess_article_tokens(or_article)
  abs = preprocess_abs_tokens(or_abs)
  stop = False
  if article is False:
    stop = True
  if abs is False:
    stop = True
  return [article, abs]

def filter_save_paris(lines, bin_path, text_path, abs_index=1, article_index=2,
                      source_min_tokens=None, source_max_tokens=None,
                      target_min_tokens=None, target_max_tokens=None,
                      vocab_path=None, makevocab=False):

  if makevocab and vocab_path is None:
    raise  ValueError("make vocab must provide vocab_path")

  writer = codecs.open(bin_path, "wb")
  ft = codecs.open(text_path, "w", "utf-8")

  if makevocab:
    vocab_counter = collections.Counter()

  valid_samples = 0
  for i, line in enumerate(lines):

    # Get the strings to write to .bin file
    # article, abstract = get_article_abs(line)
    # if article == False or abstract == False:
    #   continue

    x = get_source_target_pairs(line, source_min_tokens=source_min_tokens, source_max_tokens=source_max_tokens,
                                  target_min_tokens=target_min_tokens, target_max_tokens=target_max_tokens)
    if x is False:
      continue

    source, target = x
    st = "\t".join([source, target]) + "\n"
    ft.write(six.text_type(st))

    source = source.encode("utf-8")
    target = target.encode("utf-8")

    # Write to tf.Example
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([source])
    tf_example.features.feature['abstract'].bytes_list.value.extend([target])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

    valid_samples += 1
    # Write the vocab to file, if applicable
    if makevocab:
      if sys.version_info[0] == 3:
        source = source.decode("utf-8")
        target = target.decode("utf-8")
      art_tokens = source.split(' ')
      abs_tokens = target.split(' ')
      abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
      tokens = art_tokens + abs_tokens
      tokens = [t.strip() for t in tokens]  # strip
      tokens = [t for t in tokens if t != ""]  # remove empty
      vocab_counter.update(tokens)

  ft.close()
  writer.close()
  print("have {} samples".format(valid_samples))
  print("Finished all articles and abs writing bin file %s" % bin_path)
  print("write text articles and abs to {}\n".format(text_path))

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with codecs.open(vocab_path, 'w', "utf-8") as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        s = word + ' ' + str(count) + '\n'
        if sys.version_info[0] == 2:
          s = s.decode("utf-8")
        writer.write(s)
    print("Finished writing vocab file")

def make_bin_data(token_file_or_dir, bin_dir, vocab_dir, abs_index=1, article_index=2, ratios="0.8,0.1,0.1"):
  from os.path import join
  if os.path.exists(bin_dir) == False:
    os.makedirs(bin_dir)
  if os.path.exists(vocab_dir) == False:
    os.makedirs(vocab_dir)
  lines = []
  if os.path.isdir(token_file_or_dir):
    for filename in os.listdir(token_file_or_dir):
      lines.extend(codecs.open(join(token_file_or_dir, filename), "r", "utf-8").readlines())
  else:
    lines = codecs.open(token_file_or_dir, "r", "utf-8").readlines()
  ratio_list = [float(v) for v in ratios.split(",")]
  assert  len(ratio_list) == 3, ratio_list
  train_index = int(len(lines) * ratio_list[0])
  dev_index = int( len(lines) * (ratio_list[0] + ratio_list[1]) )
  test_index = int( len(lines) * np.sum(ratio_list) )
  train_lines = lines[0 : train_index]
  val_lines = lines[train_index : dev_index]
  test_lines = lines[dev_index : test_index]
  train_bin_path, vocab_path = join(bin_dir, "train.bin") , join(vocab_dir, "vocab.txt")
  dev_bin_path = join(bin_dir, "val.bin")
  test_bin_path = join(bin_dir, "test.bin")
  filter_save_paris(train_lines, train_bin_path, join(bin_dir, "train.txt"), makevocab=True, vocab_path=vocab_path)
  filter_save_paris(val_lines, dev_bin_path, join(bin_dir, "val.txt"))
  filter_save_paris(test_lines, test_bin_path, join(bin_dir, "test.txt"))

def sentence_ok(token_str, min_tokens=None, max_tokens=None):
  tokens = token_str.split()
  l = len(tokens)
  if min_tokens is not None and l < min_tokens:
    return False
  if max_tokens is not None and l > max_tokens:
    return False
  return True

def get_source_target_pairs(line, source_min_tokens=None, source_max_tokens=None, target_min_tokens=None, target_max_tokens=None,
                               max_overlap=None, source_index=0, target_index=1, delimeter='\t'):
  cells = line.strip().split(delimeter)
  try:
    source_str, target_str = cells[source_index], cells[target_index]
  except IndexError:
    print("source_index={}, target_index={}, error".format(source_index, target_index))
    print(line)
    return False
  source_ok = sentence_ok(source_str, source_min_tokens, source_max_tokens)
  if source_ok is False:
    return False
  target_ok = sentence_ok(target_str, target_min_tokens, target_max_tokens)
  if target_ok is False:
    return False
  return [source_str, target_str]

@click.group()
def cli():
  pass

@click.command()
@click.argument("path_or_dir")
@click.argument("write_dir")
@click.argument("set_name")
@click.option("--makevocab", is_flag=True)
@click.option("--chunk_size", default=10000)
def convert_parallel_text(path_or_dir, write_dir, set_name, makevocab=True, chunk_size=10000):
  assert  set_name in ["train", "val", "test"]
  from os.path import join
  paths = util.get_dir_or_file_path(path_or_dir)
  bin_dir = join(write_dir, "bin")
  vocab_dir = join(write_dir, "vocab")
  chunks_dir = join(write_dir, "chunked")
  if os.path.exists(bin_dir) == False:
    os.makedirs(bin_dir)
  if os.path.exists(vocab_dir) == False:
    os.makedirs(vocab_dir)
  if os.path.exists(chunks_dir) == False:
    os.makedirs(chunks_dir)
  bin_path = os.path.join(bin_dir, "{}.bin".format(set_name))
  text_path = join(bin_dir, "{}.txt".format(set_name))
  vocab_path = join(vocab_dir, "vocab.txt")
  lines = []
  for path in paths:
    lines.extend(codecs.open(path, "r", "utf-8").readlines())
  filter_save_paris(lines, bin_path, text_path, makevocab=makevocab, vocab_path=vocab_path)
  chunk_file(bin_path, chunks_dir, set_name,chunk_size = chunk_size)

def chunk_file(bin_path, chunks_dir, set_name, chunk_size=1000):
  in_file = bin_path
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(chunk_size):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def chunk_all(bin_dir, chunks_dir):
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    bin_path = os.path.join(bin_dir, "{}.bin".format(set_name))
    print ("Splitting %s data into chunks..." % bin_path)
    chunk_file(bin_path, chunks_dir, set_name)
  print ("Saved chunked data in %s" % chunks_dir)

@click.command()
@click.argument("source_path_or_dir")
@click.argument("write_dir")
@click.option("--token_dir_name", default="token", help="under {write_dir}/{token_dir_name} [None]")
@click.option("--token_file_name", default=None, help="when source is file, must provide(suffix may not .token")
@click.option("--abs_index", default=1, type=int, help="abstract index in one line[1]")
@click.option("--article_index", default=2, type=int, help="article index in one line[2]")
@click.option("--ratios", default="0.9:0.05:0.05", type=str, help="train:dev:test=0.9:0.05:0.05")
@click.option("--tokenize", is_flag=True, help="if set, tokenize")
def mak_sum_data(source_path_or_dir, write_dir, tokenize=True, token_dir_name=None, token_file_name=None, abs_index=1, article_index=2, ratios="0.9,0.05,0.05"):
  from os.path import join
  if os.path.isdir(source_path_or_dir):
    if token_dir_name is None:
      assert token_file_name != None, "must provide tokenized_file_name when given dir"
      token_path = join(write_dir, token_file_name + ".token")
  else:
    if token_file_name is None:
      token_file_name = os.path.basename(source_path_or_dir).split(".")[0]
    token_path = join(write_dir, token_file_name + ".token")

  token_path_or_dir = None
  if token_dir_name is not None:
    token_path_or_dir = os.path.join(write_dir, token_dir_name)
    if os.path.exists(token_path_or_dir) == False:
      os.makedirs(token_path_or_dir)
  else:
    token_path_or_dir = token_path
  bin_dir = join(write_dir, "bin")
  chunks_dir = join(write_dir, "chunked")
  vocab_path = join(write_dir, "vocab")
  if tokenize:
    token_file_or_dir(source_path_or_dir, token_path_or_dir)
  else:
    print("will not tokenize {}".format(source_path_or_dir))
  make_bin_data(token_path_or_dir, bin_dir, vocab_path, abs_index=abs_index, article_index=article_index, ratios=ratios)
  chunk_all(bin_dir, chunks_dir)

cli.add_command(convert_parallel_text)
cli.add_command(mak_sum_data)

if __name__ == '__main__':

  cli()