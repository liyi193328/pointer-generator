#encoding=utf-8

import sys
import os
import codecs
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

import pyltp
LTP_DATA_DIR = "/home/bigdata/software/LTP/ltp_data"
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

from pyltp import Segmentor
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型
words = segmentor.segment('元芳你怎么看')  # 分词
print ("  ".join(words))

def token_file(file_path, save_dir, save_name=None, delimiter="\t"):
  base_name = os.path.basename(file_path)
  if save_name == None:
    save_name = base_name
  save_path = os.path.join(save_dir, save_name)
  fo = codecs.open(save_path, "w", "utf-8")
  all_lines = codecs.open(file_path, "r", "utf-8").readlines()
  new_lines = []
  for i, line in enumerate(all_lines):
    t = line.strip().split(delimiter)
    new_line = []
    for every_element in t:
      tokens = segmentor.segment(every_element.strip())
      tokens_str = " ".join(tokens)
      new_line.append(tokens)
    line_str = "\t".join(new_line)
    fo.write(line_str)
  print("tokize {}, save to {}".format(file_path, save_path))
  fo.close()

def preprocess_abs(abs):
  try:
    abs = abs.split(u"：")[1]
  except Exception:
    return abs
def preprocess_article(article):
  return article

def get_art_abs(line, delimeter="\t", abs_index=1, article_index=2):
  eles = line.strip().split(delimeter)
  try:
    article, abs = eles[article_index], eles[abs_index]
  except IndexError:
    return [False, False]
  article = preprocess_article(article)
  abs = preprocess_abs(abs)
  return [article, abs]

def save_article_abs(tokenized_file_path, bin_path, text_path, abs_index=1, article_index=2, vocab_path=None, makevocab=False):

  if makevocab and vocab_path is None:
    raise  ValueError("make vocab must provide vocab_path")

  f = codecs.open(tokenized_file_path, "r", "utf-8")
  writer = codecs.open(bin_path, "wb")
  ft = codecs.open(text_path, "w", "utf-8")
  lines = f.readlines()

  if makevocab:
    vocab_counter = collections.Counter()

  for i, line in enumerate(lines):
    # Get the strings to write to .bin file
    article, abstract = get_art_abs(line)
    if article == False:
      continue
    ft.write("\t".join([abstract,article]))

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
      abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
      tokens = art_tokens + abs_tokens
      tokens = [t.strip() for t in tokens]  # strip
      tokens = [t for t in tokens if t != ""]  # remove empty
      vocab_counter.update(tokens)

  ft.close()
  writer.close()

  print("Finished all articles and abs writing bin file %s\n" % bin_path)
  print("write text articles and abs to {}".format(text_path))

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with codecs.open(vocab_path, 'w', "utf-8") as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")

def
dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

all_train_urls = "url_lists/all_train.txt"
all_val_urls = "url_lists/all_val.txt"
all_test_urls = "url_lists/all_test.txt"

cnn_tokenized_stories_dir = "cnn_stories_tokenized"
dm_tokenized_stories_dir = "dm_stories_tokenized"
finished_files_dir = "finished_files"
chunks_dir = os.path.join(finished_files_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000 # num examples per chunk, for the chunked data


def chunk_file(set_name):
  in_file = 'finished_files/%s.bin' % set_name
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


def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['train', 'val', 'test']:
    print ("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print ("Saved chunked data in %s" % chunks_dir)


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
  print "Making bin file for URLs listed in %s..." % url_file
  url_list = read_text_file(url_file)
  url_hashes = get_url_hashes(url_list)
  story_fnames = [s+".story" for s in url_hashes]
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fnames):
      if idx % 1000 == 0:
        print "Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(cnn_tokenized_stories_dir, s)):
        story_file = os.path.join(cnn_tokenized_stories_dir, s)
      elif os.path.isfile(os.path.join(dm_tokenized_stories_dir, s)):
        story_file = os.path.join(dm_tokenized_stories_dir, s)
      else:
        print "Error: Couldn't find tokenized story file %s in either tokenized story directories %s and %s. Was there an error during tokenization?" % (s, cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        # Check again if tokenized stories directories contain correct number of files
        print "Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
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

  print "Finished writing file %s\n" % out_file

  # write vocab to file
  if makevocab:
    print "Writing vocab file..."
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print "Finished writing vocab file"


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


if __name__ == '__main__':
  if len(sys.argv) != 3:
    print "USAGE: python make_datafiles.py <cnn_stories_dir> <dailymail_stories_dir>"
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
