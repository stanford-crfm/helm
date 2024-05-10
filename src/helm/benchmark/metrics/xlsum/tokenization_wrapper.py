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


# Lint as: python2, python3
"""A library for tokenizing text."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .tokenizers import BasicTokenizer


def tokenize(text, stemmer=None, tokenizer=None):
  """Tokenize input text into a list of tokens.
  
  Args:
    text: A text blob to tokenize.
    stemmer: An optional stemmer.
    tokenizer: An optional tokenizer.

  Returns:
    A list of string tokens extracted from input text.
  """
  if tokenizer is None:
    tokenizer = BasicTokenizer()

  # Convert everything to lowercase.
  text = text.lower()
  # replace punctuation and tokenize
  tokens = tokenizer(text)

  if stemmer:
    tokens = [stemmer(x) for x in tokens]

  tokens = [x for x in tokens if x]
  return tokens
