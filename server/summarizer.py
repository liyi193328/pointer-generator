from nltk import word_tokenize

from batcher import Example, Batch
minimum_summarization_length = 200


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
    return self.decoder.decode(single_batch)  # decode indefinitely (unless single_pass=True, in
    # which case deocde the dataset exactly once)

  def article_to_batch(self, article):
    abstract_sentences = ''
    example = Example(article, abstract_sentences, self.vocab, self.hps)  # Process into an Example.
    repeated_example = [example for _ in range(self.hps.batch_size)]
    return Batch(repeated_example, self.hps, self.vocab)