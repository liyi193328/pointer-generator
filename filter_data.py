#encoding=utf-8
import os
import pandas as pd
import numpy as np
import sys
import codecs
import glob
import click
import charset
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

@click.group()
def cli():
  pass

@click.command()
@click.argument("vocab_path")
@click.argument("save_path")
# @click.argument("--del_num/--no-del_num", default=True)
# @click.argument("--del_en/--no-del_en", default=True)
def filter_vocab(vocab_path, save_path, del_num=True, del_en=True):
  f = codecs.open(vocab_path, "r", "utf-8")
  fout = codecs.open(save_path, "w", "utf-8")
  word_sets = set()
  while True:
    line = f.readline()
    if line == "":
      break
    try:
      t = line.strip().split(" ")
      word, count = t[0], t[1]
    except Exception:
      import traceback
      traceback.print_exc()
      print(line)
      continue
    skip = False

    for index, char in enumerate(word):
        if charset.is_alphabet(char) or charset.is_number(char):
          print(word)
          skip = True
          break
        if charset.is_chinese_punctuation(char): ##solve 。榜样
          if len(word) > 1:
            print(word)
            skip = True
            break

    if skip is False:
      wst = "\t".join([word, str(count)]) + "\n"
      fout.write(wst)
  f.close()
  fout.close()

@click.command()
@click.argument("path")
@click.argument("stat_path")
@click.argument("save_path", type=str)
@click.option("--min_abs_chars", type=int, default=0, help="abs min char[0]")
@click.option("--max_abs_chars", type=int, default=int(2e5), help="max abs char[2e5]")
@click.option("--title_index", type=int, default=1, help="title index in one line[1]")
@click.option("--abs_index", type=int, default=2, help="abs index in one line[1]")
@click.option("--article_index", type=int, default=3, help="article index in one line[1]")
@click.option("--cancel_unk_label", is_flag=True, help="whether cancel null label")
@click.option("--abs_max_cov_sents", type=int, default=None, help="abstract very high sim to article 's max sents[100]")
def filter_stat(path, stat_path, save_path, label_index=0, title_index=1, abs_index=2, article_index=3,
                min_abs_chars=0, max_abs_chars=int(2e5), abs_max_cov_sents=None, cancel_unk_label=False):
  len_stat = []
  use_samples =  0
  fin = codecs.open(path, "r", "utf-8")
  fo = codecs.open(save_path, "w", "utf-8")
  total_abs_sent_num , cov_abs_sent_num = 0 , 0
  cov_skip_abs = 0
  i = 0
  no_label_num = 0
  abs_len_filter_num = 0
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
        label = t[label_index]
      title = t[title_index]
      abstract = t[abs_index]
      article = t[article_index]
    except IndexError:
      continue
    abstract = make_sum_data.preprocess_abs_text(abstract)
    if len(abstract) > max_abs_chars or len(abstract) < min_abs_chars:
      abs_len_filter_num += 1
      continue
    abs_sents = SentenceSplitter.split(abstract)
    total_abs_sent_num += len(abs_sents)
    cov_num = 0
    for sent in abs_sents:
      if sent in article:
        cov_num += 1
    labeled = False
    if label is not None and "null" in label:
      labeled = True
      no_label_num +=1
    cov_abs_sent_num += cov_num
    if len(abs_sents) == cov_num or cov_num > abs_max_cov_sents:
      cov_skip_abs += 1
      continue
    if cancel_unk_label is False or labeled is True:
      use_samples += 1
      len_stat.append([len(article), len(abstract)])
      fo.write("\t".join([title, abstract, article]) + "\n")

  print("filter docs with abs len not in range[{},{})".format(abs_len_filter_num))
  print("skip {} docs with high coverage in article".format(cov_skip_abs))
  print("{} docs have no label".format(no_label_num))
  print("use:all = {}:{}".format(use_samples, i))
  print("abs-cov:all = {}:{}".format(cov_abs_sent_num, total_abs_sent_num))
  len_df = pd.DataFrame(len_stat, columns=["art_len", "abs_len"])
  print("use sample len stat")
  print(len_df.describe())
  len_df.to_csv(stat_path, index=False)
  fo.close()

cli.add_command(filter_vocab)
cli.add_command(filter_stat)

if __name__ == "__main__":
  cli()


