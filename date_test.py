#encoding=utf-8

import codecs
import batcher
import data
from collections import namedtuple
bin_path = "/home/bigdata/active_project/run_tasks/query_rewrite/stable/stable_single/copy_data_format/chunked/train*"
vocab_path = "/home/bigdata/active_project/run_tasks/text_sum/data/vocab/vocab.txt"
data_generater = data.example_generator(bin_path, single_pass=True)


def test_text(path):
  f = codecs.open(path, "r", "utf-8")
  lines = f.readlines()
  print(lines[0])
  print(type(lines[0]))

def test_batcher():
  HPS = namedtuple("HPS", ["batch_size", "mode"])
  hps = HPS(5, "train")
  bx = batcher.Batcher(bin_path,vocab_path,hps, single_pass=True)
  input_gen = bx.text_generator(data.example_generator(bx._data_path, bx._single_pass))
  cnt = 0
  while True:
    x = input_gen.next()
    print(x[0].decode("utf-8"))
    print(x[1].decode("utf-8"))
    cnt += 1
    if cnt > 10:
      break

if __name__ == "__main__":
  test_batcher()