# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Language specific tokenization / word segmenter classes. Uses 
    some code fragments from bert tokenizer with few modifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unicodedata
import six
from functools import partial

def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")

def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens

def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False

def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False

def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False

class BasicTokenizer(object):
  """Runs basic tokenization."""

  def __call__(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text. skips over the punctuation"""
    output = []
    chars = list(text)
    i = 0
    start_new_word = True
    
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1
    output = ["".join(x) for x in output]

    return output


  def tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

class ThaiTokenizer(object):
    """See https://pythainlp.github.io/docs/2.0/api/tokenize.html"""
    def __init__(self):
        from pythainlp.tokenize import word_tokenize
        self.tokenizer = partial(word_tokenize, engine="newmm")

    def __call__(self, text):
        return self.tokenizer(text)

class ChineseTokenizer(object):
    
	def __init__(self):
		import jieba
		self.tokenizer = jieba
		self.tokenizer.initialize()

	def __call__(self, text):
		return list(self.tokenizer.cut(text))

class JapaneseTokenizer(object):
    
	def __init__(self):
		from fugashi import Tagger
		self.tokenizer = Tagger('-O wakati -b 50000')
		
	def __call__(self, text):
		return whitespace_tokenize(self.tokenizer.parse(text))

class BurmeseTokenizer(object):
  """Implementation taken from https://dl.acm.org/doi/fullHtml/10.1145/3325885"""

  def __init__(self):
    self.X = chr (0x103a)
    self.T = chr (0x1037)
    self.STACK = chr (0x1039)
    self.TX = self.T + self.X
    self.XT = self.X + self.T

    self.DEP = set ([chr(0x102b + x) for x in range (8)])
    self.DEP |= set ([chr(0x1036 + x) for x in range (3)])
    self.DEP |= set ([chr(0x103b + x) for x in range (4)])
    self.DEP.add(self.X)

  def __call__(self, text):
    try:
      text = text.replace(chr(0x200c), ' ')
      text = text.replace(self.TX, self.XT)
      text = list(''.join(text.lower().strip().split()))
      
      ### generate basic units
      text.reverse ()
      for i in range(len(text) - 1): 
        if text[i][0] in self.DEP:
          text[i + 1] += text[i] 
          text[i] = ''

      text.reverse()

      ### attach asat units (len < 4 to avoid au)
      text = ' '.join(text).split()
      for i in range (1, len(text)):
        if self.X in text[i] and len(text[i]) < 4:
          text[i - 1] += text[i]
          text[i] = ''

      ### glue stacked units
      text = ' '.join(text).split()
      text = ' '.join(text).replace(' '+ self.STACK + ' ', self.STACK)
    except:
      pass

    return text.split()


LANG2TOKENIZER = {
	"thai": ThaiTokenizer,
	"chinese": ChineseTokenizer,
	"japanese": JapaneseTokenizer,
  "burmese": BurmeseTokenizer
}

