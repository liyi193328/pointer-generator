#encoding=utf-8

from batcher import Example, Batch


import os
USER_ROOT = os.path.expanduser("~")
DEFAULT_LTP_DATA_DIR=os.path.join(USER_ROOT, "software/LTP/ltp_data")
LTP_DATA_DIR = os.environ.get("LTP_DATA_DIR",DEFAULT_LTP_DATA_DIR)
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
try:
  import pyltp
  from pyltp import Segmentor
  from pyltp import SentenceSplitter
  segmentor = Segmentor()  # 初始化实例
  segmentor.load(cws_model_path)  # 加载模型
except ImportError:
  pass

minimum_summarization_length = 2

def word_tokenize(s):
  tokens = segmentor.segment(s.encode("utf-8"))
  return tokens

class Summarizer():
  def __init__(self, decoder, vocab, hps):
    self.decoder = decoder
    self.vocab = vocab
    self.hps = hps

  def summarize(self, input_article, tokenized=False):
    if not tokenized:
      tokenized_article = ' '.join(word_tokenize(input_article))
    else:
      tokenized_article = input_article
    if len(tokenized_article.split(" ")) < minimum_summarization_length:
      return input_article

    single_batch = self.article_to_batch(tokenized_article)
    return self.decoder.decode(single_batch, beam_nums_to_return=10)  # decode indefinitely (unless single_pass=True, in
    # which case deocde the dataset exactly once)

  def article_to_batch(self, article):
    abstract_sentences = ''
    example = Example(article, abstract_sentences, self.vocab, self.hps)  # Process into an Example.
    repeated_example = [example for _ in range(self.hps.batch_size)]
    return Batch(repeated_example, self.hps, self.vocab)