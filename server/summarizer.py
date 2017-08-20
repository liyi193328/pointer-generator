#encoding=utf-8

from batcher import Example, Batch

import pyltp
import os

LTP_DATA_DIR = os.environ.get("LTP_DATA_DIR","/home/bigdata/software/LTP/ltp_data")
cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')

from pyltp import Segmentor
from pyltp import SentenceSplitter
segmentor = Segmentor()  # 初始化实例
segmentor.load(cws_model_path)  # 加载模型

minimum_summarization_length = 4

def word_tokenize(s):
  tokens = segmentor.segment(s.encode("utf-8"))
  return tokens

class Summarizer():
  def __init__(self, decoder, vocab, hps):
    self.decoder = decoder
    self.vocab = vocab
    self.hps = hps

  def summarize(self, input_article):
    if len(input_article) < minimum_summarization_length:
      return input_article
    tokenized_article = ' '.join(word_tokenize(input_article))
    single_batch = self.article_to_batch(tokenized_article)
    return self.decoder.decode(single_batch, beam_nums_to_return=10)  # decode indefinitely (unless single_pass=True, in
    # which case deocde the dataset exactly once)

  def article_to_batch(self, article):
    abstract_sentences = ''
    example = Example(article, abstract_sentences, self.vocab, self.hps)  # Process into an Example.
    repeated_example = [example for _ in range(self.hps.batch_size)]
    return Batch(repeated_example, self.hps, self.vocab)