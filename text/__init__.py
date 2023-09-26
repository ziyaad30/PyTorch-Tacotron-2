import re
from text import cleaners
import torch
from tokenizers import Tokenizer


tokenizer = Tokenizer.from_file('./bert/tokenizer.json')
vocab = tokenizer.get_vocab()
symbols = sorted(vocab, key=lambda x: vocab[x])


def text_to_sequence(text, cleaner_names):
    text = _clean_text(text, cleaner_names)
    text = text.replace(' ', '[SPACE]')
    output = tokenizer.encode(text)
    print(output.tokens)
    print(output.ids)
    sequence_to_text(output.tokens)
    return output.ids


def sequence_to_text(tokens):
    txt = ''.join(tokens)
    txt = txt.replace('[SPACE]', ' ')
    print(txt)


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text
