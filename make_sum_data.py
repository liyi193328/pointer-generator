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


def token_file(file_path, token_path_or_handle, abs_index=1, article_index=2, delimiter="\t"):
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

def token_file_or_dir(file_or_dir, token_path_or_dir, article_index = 2, delimiter="\t", filters=None, pnums=MP.cpu_count() - 1):
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

def preprocess_abs_tokens(abs, article=None, max_substring_sents=0):

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

def preprocess_article_tokens(article, min_tokens=300, max_tokens=None, token_split=" "):
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
    print("article is False")
    print(or_article)
    stop = True
  if abs is False:
    print("abs is False")
    print(or_abs)
    stop = True
  return [article, abs]

def save_article_abs(lines, bin_path, text_path, abs_index=1, article_index=2, vocab_path=None, makevocab=False):

  if makevocab and vocab_path is None:
    raise  ValueError("make vocab must provide vocab_path")

  writer = codecs.open(bin_path, "wb")
  ft = codecs.open(text_path, "w", "utf-8")

  if makevocab:
    vocab_counter = collections.Counter()

  valid_samples = 0
  for i, line in enumerate(lines):
    # Get the strings to write to .bin file
    article, abstract = get_article_abs(line)
    if article == False or abstract == False:
      continue
    art_abs_str = "\t".join([abstract, article]) + "\n"
    ft.write(six.text_type(art_abs_str))

    article = article.encode("utf-8")
    abstract = abstract.encode("utf-8")

    # Write to tf.Example
    tf_example = example_pb2.Example()
    tf_example.features.feature['article'].bytes_list.value.extend([article])
    tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
    tf_example_str = tf_example.SerializeToString()
    str_len = len(tf_example_str)
    writer.write(struct.pack('q', str_len))
    writer.write(struct.pack('%ds' % str_len, tf_example_str))

    valid_samples += 1
    # Write the vocab to file, if applicable
    if makevocab:
      if sys.version_info[0] == 3:
        article = article.decode("utf-8")
        abstract = abstract.decode("utf-8")
      art_tokens = article.split(' ')
      abs_tokens = abstract.split(' ')
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
  save_article_abs(train_lines, train_bin_path, join(bin_dir, "train.txt"), makevocab=True, vocab_path=vocab_path)
  save_article_abs(val_lines, dev_bin_path, join(bin_dir, "val.txt"))
  save_article_abs(test_lines, test_bin_path, join(bin_dir, "test.txt"))

def chunk_file(bin_path, chunks_dir, set_name):
  in_file = bin_path
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
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
    chunk_file(bin_path,chunks_dir, set_name)
  print ("Saved chunked data in %s" % chunks_dir)

@click.command()
@click.argument("source_path_or_dir")
@click.argument("write_dir")
@click.option("--token_dir_name", default="token", help="under {write_dir}/{token_dir_name} [None]")
@click.option("--token_file_name", default=None, help="when source is file, must provide(suffix may not .token")
@click.option("--abs_index", default=1, type=int, help="abstract index in one line[1]")
@click.option("--article_index", default=2, type=int, help="article index in one line[2]")
@click.option("--ratios", default="0.8,0.1,0.1", type=str, help="train:dev:test=0.8:0.1:0.1")
def mak_sum_data(source_path_or_dir, write_dir, token_dir_name=None, token_file_name=None, abs_index=1, article_index=2, ratios="0.8,0.1,0.1"):
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
  token_file_or_dir(source_path_or_dir, token_path_or_dir)
  make_bin_data(token_path_or_dir, bin_dir, vocab_path, abs_index=abs_index, article_index=article_index, ratios=ratios)
  chunk_all(bin_dir, chunks_dir)


def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print ("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print ("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print ("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print ("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print ("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))


def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s)
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(file):
  lines = read_text_file(file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def write_to_bin(url_file, out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print ("Making bin file for URLs listed in %s..." % url_file)
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print ("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
        story_file = os.path.join(cnn_tokenized_stories_dir, s)
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_tokenized_stories_dir, s)
      else:
        print ("Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
        # Check again if tokenized stories directories contain correct number of files
        print ("Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir))
        check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
        check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s and %s contain correct number of files but story file %s found in neither." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir, s))

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print ("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print ("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print ("Finished writing vocab file")


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


def cnn_dayily_main():
  if len(sys.argv) != 3:
    print ("USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>")
    sys.exit()
  cnn_stories_dir = sys.argv[1]
  dm_stories_dir = sys.argv[2]

  # Check the stories directories contain the correct number of .story files
  check_num_stories(cnn_stories_dir, num_expected_cnn_stories)
  check_num_stories(dm_stories_dir, num_expected_dm_stories)

  # Create some new directories
  if not os.path.exists(cnn_tokenized_stories_dir): os.makedirs(cnn_tokenized_stories_dir)
  if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(cnn_stories_dir, cnn_tokenized_stories_dir)
  tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(all_test_urls, os.path.join(finished_files_dir, "test.bin"))
  write_to_bin(all_val_urls, os.path.join(finished_files_dir, "val.bin"))
  write_to_bin(all_train_urls, os.path.join(finished_files_dir, "train.bin"), makevocab=True)

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()

if __name__ == '__main__':

  mak_sum_data()