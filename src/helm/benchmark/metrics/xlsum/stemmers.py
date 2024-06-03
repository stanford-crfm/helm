from nltk.stem import SnowballStemmer
from functools import partial

class HindiStemmer(object):
  """See http://computing.open.ac.uk/Sites/EACLSouthAsia/Papers/p6-Ramanathan.pdf"""
  def __init__(self):
    self.suffixes = {
        1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
        2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
        3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
        4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
        5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
    }

  def __call__(self, token):
    for L in 5, 4, 3, 2, 1:
      if len(token) > L + 1:
          for suf in self.suffixes[L]:
              if token.endswith(suf):
                  return token[:-L]
    return token

class BengaliStemmer(object):
  """See https://github.com/abhik1505040/bengali-stemmer"""
  def __init__(self):
    from bengali_stemmer.rafikamal2014 import RafiStemmer
    self.stemmer = RafiStemmer()
  
  def __call__(self, token):
    return self.stemmer.stem_word(token)

class TurkishStemmer(object):
  """See https://github.com/otuncelli/turkish-stemmer-python"""
  def __init__(self):
    from TurkishStemmer import TurkishStemmer
    self.stemmer = TurkishStemmer()

  def __call__(self, token):
    return self.stemmer.stem(token)

class NLTKStemmer(object):
  def __init__(self, lang, ignore_stopwords):
    ignore_stopwords = False if lang == "porter" else ignore_stopwords
    self.stemmer = SnowballStemmer(lang, ignore_stopwords)

  def __call__(self, token):
    return self.stemmer.stem(token)

LANG2STEMMER = {
    "bengali": BengaliStemmer,
    "turkish": TurkishStemmer,
    "hindi": HindiStemmer,
}
LANG2STEMMER.update(
    {k: partial(NLTKStemmer, k, ignore_stopwords=True) \
        for k in SnowballStemmer.languages}
)




