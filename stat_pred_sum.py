#encoding=utf-8

import os
import codecs
import glob
import click
import argparse

@click.group()
def cli():
  pass

@click.command()
@click.argument("file_path")
def stat_one_file(file_path):
  total_coverage_sents = 0
  total_sents = 0
  print("reading {}".format(file_path))
  with codecs.open(file_path, "r", "utf-8") as f:
    while True:
      coverage_sents = 0
      infer_sents = 0
      article = f.readline()
      if not article:
        break
      article = article.strip()
      ref_sents = []
      f.readline()
      while True:
        line = f.readline()
        if line == "\n":
          break
        ref_sents.append(line.strip())
      decoded_sents = []
      while True:
        line = f.readline()
        if line == "" or line == "\n":
          break
        decoded_sents.append(line.strip())
      infer_sents += len(decoded_sents)
      for sent in decoded_sents:
        if sent in article:
          coverage_sents += 1
      total_coverage_sents += coverage_sents
      total_sents += infer_sents
  print(total_sents, total_coverage_sents)
  return [total_sents, total_coverage_sents]

@click.command()
@click.argument("pred_dir")
@click.pass_context
def stat_pred_sum( ctx, pred_dir, save_result_path=None):
  files = [os.path.join(pred_dir, file) for file in os.listdir(pred_dir)]
  total_sents = 0
  total_coverage = 0
  for path in files:
    print(path)
    sents, coverages = ctx.invoke(stat_one_file, file_path=path)
    total_sents += sents
    total_coverage += coverages
  print("substring/total: {}/{} = {}".format(total_coverage, total_sents, float(total_coverage)/total_sents))

cli.add_command(stat_one_file)
cli.add_command(stat_pred_sum)

if __name__ == "__main__":
  # parser = argparse.ArgumentParser()
  # parser.add_argument("pred_dir", type=str)
  # args = parser.parse_args()
  # stat_pred_sum(args.pred_dir)
  cli()