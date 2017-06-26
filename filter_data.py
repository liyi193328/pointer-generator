#encoding=utf-8
import os
import pandas as pd
import numpy as np
import sys
import codecs
import glob
import click
import make_sum_data
from pyltp import SentenceSplitter

def write_list_to_file(d, file_path, verb=True):
  assert type(d) == list
  if os.path.exists(os.path.dirname(file_path)) == False:
    os.makedirs(os.path.dirname(file_path))
  f = codecs.open(file_path, "w", "utf-8")
  for x in d:
    x = str(x)
    f.write(x.strip() + "\n")
  f.close()
  if verb:
    print("write to {}".format(file_path))

@click.command()
@click.argument("path")
@click.argument("stat_path")
@click.argument("save_path", type=str)
@click.option("--title_index", type=int, default=1, help="title index in one line[1]")
@click.option("--abs_index", type=int, default=2, help="abs index in one line[1]")
@click.option("--article_index", type=int, default=3, help="article index in one line[1]")
def filter_stat(path, stat_path, save_path, label_index=0, title_index=1, abs_index=2, article_index=3, abs_max_cov_sents=1):
  len_stat = []
  use_samples =  0
  fin = codecs.open(path, "r", "utf-8")
  fo = codecs.open(save_path, "w", "utf-8")
  total_abs_sent_num , cov_abs_sent_num = 0 , 0
  cov_skip_abs = 0
  i = 0
  no_label_num = 0
  while True:
    line = fin.readline()
    if not line:
      break
    if (i+1) % int(1e4) == 0:
      print("finished {}...".format(i))
    i += 1
    t = line.strip().split("\t")
    try:
      label = None
      if label_index is not None:
        label = t[label]
      title = t[title_index]
      abstract = t[abs_index]
      article = t[article_index]
    except IndexError:
      continue
    abstract = make_sum_data.preprocess_abs_text(abstract)
    abs_sents = SentenceSplitter.split(abstract)
    len_stat.append( [len(article), len(abstract)] )
    total_abs_sent_num += len(abs_sents)
    cov_num = 0
    for sent in abs_sents:
      if sent in article:
        cov_num += 1
    if label is not None and "null" in label:
      no_label_num +=1
    cov_abs_sent_num += cov_num
    if len(abs_sents) == cov_num or cov_num > abs_max_cov_sents:
      cov_skip_abs += 1
      continue
    if label is None or "null" not in label:
      use_samples += 1
      fo.write("\t".join([title, abstract, article]) + "\n")

  print("skip {} docs with high coverage in article".format(cov_skip_abs))
  print("{} docs have no label".format(no_label_num))
  print("use:all = {}:{}".format(use_samples, i))
  print("abs-cov:all = {}:{}".format(cov_abs_sent_num, total_abs_sent_num))
  len_df = pd.DataFrame(len_stat, columns=["art_len", "abs_len"])
  print(len_df.describe())
  len_df.to_csv(stat_path, index=False)

  fo.close()
if __name__ == "__main__":
  filter_stat()


