""" from https://github.com/keithito/tacotron """
import re
from text import cleaners
from text.symbols import symbols

# load phonemizer
from phonemizer.backend import EspeakBackend

# Mappings from symbol to numeric ID and vice versa:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r'(.*?)\{(.+?)\}(.*)')


def phoneme_text(text):
    backend = EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=False, punctuation_marks=';:,.!?¡¿—…"«»“”()', language_switch='remove-flags')
    text = backend.phonemize([text], strip=True)[0]
    return text.strip()


def phoneme_to_sequence(text, cleaner_names):
    sequence = []
    text = _clean_text(text, cleaner_names)
    
    # print(f'CLEANED: {text}')
    
    text = phoneme_text(text)
    
    # print(f' PHONED: {text}')
    
    sequence += _symbols_to_sequence(text)
    
    # t = sequence_to_text(sequence)
    # print(f'FROM SEQUENCE TO TEXT ---> {text}')

    return sequence


def text_to_sequence(text, cleaner_names):
    sequence = []
    print('\n')
    # First clean text
    text = _clean_text(text, cleaner_names)
    print(f" CLEANED TEXT ----> {text}\n")
    
    # Phonemize text
    text = phoneme_text(text)
    print(f"PHONEMED TEXT ----> {text}\n")

    # Check for curly braces and treat their contents as ARPAbet:
    sequence += _symbols_to_sequence(text)
    
    t = sequence_to_text(sequence)
    print(f'SEQUENCE TEXT ----> {text}\n')

    return sequence


def sequence_to_text(sequence):
  '''Converts a sequence of IDs back to a string'''
  result = ''
  for symbol_id in sequence:
    if symbol_id in _id_to_symbol:
      s = _id_to_symbol[symbol_id]
      # Enclose ARPAbet back in curly braces:
      if len(s) > 1 and s[0] == '@':
        s = '{%s}' % s[1:]
      result += s
  return result.replace('}{', ' ')


def _clean_text(text, cleaner_names):
  for name in cleaner_names:
    cleaner = getattr(cleaners, name)
    if not cleaner:
      raise Exception('Unknown cleaner: %s' % name)
    text = cleaner(text)
  return text


def _symbols_to_sequence(symbols):
  return [_symbol_to_id[s] for s in symbols if _should_keep_symbol(s)]


def _arpabet_to_sequence(text):
  return _symbols_to_sequence(['@' + s for s in text.split()])


def _should_keep_symbol(s):
  return s in _symbol_to_id and s != '_' and s != '~'
