#encoding=utf-8

import os
import sys
import tensorflow as tf
import click
import codecs
from server.run_server import summarizer

FLAGS = tf.app.flags.FLAGS

def make_infer():
  lines = codecs.open(FLAGS.infer_source_path, "r", "utf-8").readlines()
  if FLAGS.infer_save_path is None:
    fout = sys.stdout
  else:
    fout = codecs.open(FLAGS.infer_save_path, "w", "utf-8")
  for i, source in enumerate(lines):
    source = source.strip()
    decode_output_list = summarizer.summarize(source, tokenized=True)
    fout.write(source + "\n")
    for i, decode_output in enumerate(decode_output_list):
      fout.write(decode_output.strip() + "\n")
    fout.write("\n")

if __name__ == "__main__":
  make_infer()