import os
import string
from typing import List, Set, Optional

# Adapted from https://github.com/stanfordnlp/chirpycardinal/blob/main/chirpy/core/offensive_classifier/offensive_classifier.py
# And https://github.com/stanfordnlp/chirpycardinal/blob/main/chirpy/core/util.py

# SPECIAL_CHARS is copied from profanity_checker.py
SPECIAL_CHARS = "!\"#$%&()+,./:;<=>?[\\]^_`{|}~"

# Offensive phrases which are in offensive_phrases_preprocessed.txt, that you want to remove
# Enter them exactly as they appear in offensive_phrases_preprocessed.txt
REMOVE_FROM_BLACKLIST = set(['god', 'gods', 'ginger', 'gay', 'lesbian', 'lesbians', 'balls', 'crabs', 'dammit', 'damn',
                           'darn', 'dirty', 'murder', 'omg', 'organ', 'organs', 'queer', 'sandler', 'sandlers',
                           'rum', 'strips', 'pots', 'sexes', 'psycho', 'beaver', 'willy', 'mick', 'aryan', 'dink',
                           'crackers', 'ugly', 'trashy', 'crappy', 'shitty', 'sucks', 'stupid', 'tart', 'cracker', 'crackers'])

# Offensive phrases which aren't in offensive_phrases_preprocessed.txt, that you want to add
# These should be lowercase, but they can contain punctuation
# You might want to add both singular and plural versions


ADD_TO_BLACKLIST = set(['pornhub', 'penises', 'suicide', 'suicides', 'marijuana', 'poo', 'blood', 'stripper',
                        'strippers', 'sexually', 'my balls', 'his balls', 'no balls', 'pornographic', 'abortion',
                        'torture', 'tortures', 'killer', 'killing', 'talk dirty', 'dirty talk', 'talking dirty',
                        'dirty talking', 'prostitution', 'urinate', 'mating', 'feces', 'swine', 'excreted' ,'excrete', 'blackface',
                        'abusive', 'big gay al', 'big gay boat ride', 'misogyny'])

# Inoffensive phrases that we don't want to classify as offensive, even though they contain a blacklisted phrase.
# For example we don't want to classify 'kill bill' as offensive, but we don't want to remove 'kill' from the blacklist.
# These phrases will be removed from text before the text is checked for offensive phrases.
# These phrases should be lowercase.
# These phrases might be said by the bot, so they need to be inoffensive enough for us to say.
WHITELIST_PHRASES = set(['kill bill', 'beavis and butt head', 'beavis and butt-head', 'beavis and butthead',
                         'kill a mockingbird', 'killer mockingbird', 'bloody mary', 'mick jagger', 'the mick',
                         "dick's sporting goods", 'lady and the tramp', 'jackson pollock', 'on the basis of sex',
                         'sex and the city', 'sex education', 'willy wonka and the chocolate factory', 'lady and the tramp',
                         'suicide squad', "hell's kitchen", 'hells kitchen', 'jane the virgin', 'scared the hell',
                         'harry potter and the half blood prince', 'to kill a mockingbird', 'rambo last blood',
                         'shits creek', 'shit\'s creek', 'looney tunes', 'sniper', 'punky brewster',
                         'the good the bad and the ugly', 'pee wee herman', 'the ugly dachshund', 'xxx tentacion',
                         'lil uzi vert', 'lil uzi', 'young blood', 'chicken pot pie', 'pot roast', 'pop tarts',
                         'they suck', 'he\'s sexy', 'she\'s sexy', 'vegas strip', 'hell comes to frogtown',
                         'dick van dyke', 'blood and bullets', 'blood prison', 'dick powell', 'comic strip', 'comic strips'])



preprocessed_blacklist_file = os.path.join(os.path.dirname(__file__), 'blacklist.txt')

def load_text_file(filepath):
    """Loads a text file, returning a set where each item is a line of the text file"""
    lines = set()  # set of strings
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                lines.add(line)
    return lines

def remove_punc(text: str, keep: List[str] = [], replace_with_space: List[str] = ['-', '/']):
    """
    Removes all Unicode punctuation (this is more extensive than string.punctuation) from text.
    Most punctuation is replaced by nothing, but those listed in replace_with_space are replaced by space.
    Solution from here: https://stackoverflow.com/questions/11066400/remove-punctuation-from-unicode-formatted-strings/11066687#11066687
    Inputs:
        keep: list of strings. Punctuation you do NOT want to remove.
        replace_with_space: list of strings. Punctuation you want to replace with a space (rather than nothing).
    """
    punc_table = {codepoint: replace_str for codepoint, replace_str in PUNC_TABLE.items() if chr(codepoint) not in keep}
    punc_table = {codepoint: replace_str if chr(codepoint) not in replace_with_space else ' ' for codepoint, replace_str
                  in punc_table.items()}
    text = text.translate(punc_table)
    text = " ".join(text.split()).strip()  # Remove any double-whitespace
    return text

def get_ngrams(text, n):
    """
    Returns all ngrams that are in the text.
    Inputs:
      text: string, space-separated
      n: int
    Returns:
      list of strings (each is a ngram, space-separated)
    """
    tokens = text.split()
    return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-(n-1))]  # list of str

def contains_phrase(text: str, phrases: Set[str],
                    lowercase_text: bool = True, lowercase_phrases: bool = True,
                    remove_punc_text: bool = True, remove_punc_phrases: bool = True,
                    max_phrase_len: Optional[int] = None):
    """
    Checks whether text contains any phrase in phrases.
    By default, the check is case blind (text and phrases are lowercased before checking).
    This function is optimized for when phrases is a large set (for each eligible ngram in text, we check if it's in
    phrases). If phrases is small and text is long, it might be more efficient to check, for each phrase, whether it's
    in text.
    Note, a phrase is "in" text iff the sequence of words in phrase is a subsequence of the sequence of words in text.
        if phrase="name" and text="my name is julie", result=True
        if phrase="name" and text="i collect ornaments", result=False
        if phrase="your name" and text="what is your name", result=True
        if phrase="your name" and text="my name is julie what is your favorite color", result=False
    Inputs:
        text: string, space-separated.
        phrases: set of strings, space-separated.
        log_message: If not empty, a str to be formatted with (text, phrase) that gives an informative log message when
            the result is True.
        lowercase_text: if True, text will be lowercased before checking.
        lowercase_phrases: if True, all phrases will be lowercased before checking.
        remove_punc_text: if True, punctuation will be removed from text before checking.
        remove_punc_phrases: if True, punctuation will be removed from phrases before checking.
        max_phrase_len: max len (in words) of the longest phrase in phrases. If None, this will be computed from phrases
    """
    if lowercase_text:
        text = text.lower()
    if lowercase_phrases:
        phrases = set([p.lower() for p in phrases])
    if remove_punc_text:
        text = remove_punc(text)
    if remove_punc_phrases:
        phrases = set([remove_punc(p) for p in phrases])
    if max_phrase_len is None:
        max_phrase_len = max((len(phrase.split()) for phrase in phrases))

    # For each ngram in text (n=1 to max_phrase_len), check whether the ngram is in phrases.
    # Even if phrases is very large, this fn is efficient because we only look up the (relatively few) ngrams in text,
    # rather than iterating through all of phrases. Note phrases is a set so checking membership is fast.
    length = len(text.split())
    for n in range(1, min(max_phrase_len, length)+1):
        ngrams = get_ngrams(text, n)
        for ngram in ngrams:
            if ngram in phrases:
                return True
    return False

def initialize_blacklist():
    """
    Load the preprocessed blacklist from file. The blacklist is lowercase and already contains alternative versions
    of offensive phrases (singulars, plurals, variants with and without punctuation).
    """
    blacklist = load_text_file(preprocessed_blacklist_file)  # set of lowercase strings
    blacklist = blacklist.difference(REMOVE_FROM_BLACKLIST)
    blacklist = blacklist.union(ADD_TO_BLACKLIST)
    blacklist_max_len = max({len(phrase.split()) for phrase in blacklist})
    return blacklist, blacklist_max_len

def contains_offensive(text: str, blacklist: Set[str], blacklist_max_len: int) -> bool:
        # Lowercase
        text = text.lower().strip()

        # Remove whitelisted phrases from text
        for whitelisted_phrase in WHITELIST_PHRASES:
            if whitelisted_phrase in text:
                text = text.replace(whitelisted_phrase, '').strip()

        # List of variants of text to check
        texts = set()

        # Remove special characters the same way the Amazon code does (leaving * and ' in)
        texts.add(text.translate({ord(p): '' for p in SPECIAL_CHARS}))

        # Remove all string.punctuation, replacing with ''.
        # Unlike the Amazon code, this will catch things like "pissin'".
        # "pissin" and "pissing" are in our blacklist, but "pissin'" is not.
        # texts.add(text.translate({ord(p): '' for p in string.punctuation}))

        # Remove all string.punctuation, replacing with ' '.
        # This will catch things like "fuck-day" or "shit's" where we have an offensive word ("fuck", "shit") connected
        # via punctuation to a non-offensive word ("day", "s"), and the compound is not in our blacklist.
        texts.add(' '.join(text.translate({ord(p): ' ' for p in string.punctuation}).split()))

        # Also check the original text with no punctuation removed
        # This will catch things like "a$$" which are on our blacklist.
        # However, it won't catch "a$$" if it occurs next to non-whitespace e.g. "I love a$$."
        texts.add(text)

        # Check all the variants
        for text in texts:
            if contains_phrase(text, blacklist, lowercase_text=False, lowercase_phrases=False,
                               remove_punc_text=False, remove_punc_phrases=False, max_phrase_len=blacklist_max_len):
                return True

        # dealing with false positives for "He'll"
        if contains_phrase(text.translate({ord(p): '' for p in string.punctuation}), blacklist - {'hell'}, lowercase_text=False, lowercase_phrases=False,
                           remove_punc_text=False, remove_punc_phrases=False, max_phrase_len=blacklist_max_len):
                return True
        return False


    