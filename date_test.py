#encoding=utf-8

import codecs
import batcher
import data

bin_path = "/home/bigdata/active_project/run_tasks/text_sum/data/chunked/train_*"
vocab_path = "/home/bigdata/active_project/run_tasks/text_sum/data/vocab/vocab.txt"
data_generater = data.example_generator(bin_path, single_pass=True)

# samples = 10
# cnt = 0
# for x in data_generater:
#   print(x)
#   cnt += 1
#   if cnt >= samples:
#     break


def test_text(path):
  f = codecs.open(path, "r", "utf-8")
  lines = f.readlines()
  print(lines[0])
  print(type(lines[0]))

def test_batcher():
  hps = {
    "batch_size": 30
  }
  bx = batcher.Batcher(bin_path,vocab_path,hps, single_pass=True)
  input_gen = bx.text_generator(data.example_generator(bx._data_path, bx._single_pass))
  x = input_gen.next()
  print(x)
if __name__ == "__main__":
  # test_text("/home/bigdata/active_project/run_tasks/text_sum/data/bin/test.txt")
  test_batcher()