---
title: Perturbations
---
# Perturbations

## cleva_perturbation

### CLEVAMildMixPerturbation()

CLEVA robustness perturbation that composes several perturbations.

#### chinese_typos_perturbation = ChineseTyposPerturbation(0.05)instance-attribute

#### synonym_perturbation = ChineseSynonymPerturbation(0.3)instance-attribute

### ChineseGenderPerturbation(mode: str, prob: float, source_class: str, target_class: str)

Individual fairness perturbation for Chinese gender terms and pronouns.

mode (`str`) –

The mode of the gender perturbation, must be one of "terms" or "pronouns".

prob (`float`) –

Probability of substituting a word in the source class with a word in the target class given that a substitution is available.

source_class (`str`) –

The source gender that will be substituted with the target gender. If mapping_file_path is provided, the source class must be one of the genders in it. If not, it must be exactly one of`male`,`female`, and`neutral`. Case-insensitive.

target_class (`str`) –

Same as the source class, but for the target gender.

| Parameters: |
| --- |

### ChinesePersonNamePerturbation(prob: float, source_class: Dict[str, str], target_class: Dict[str, str], preserve_gender: bool = True)

Individual fairness perturbation for Chinese person names.

Code adopted from [https://github.com/PacificAI/medhelm/blob/main/src/helm/benchmark/augmentations/person_name_perturbation.py](https://github.com/PacificAI/medhelm/blob/main/src/helm/benchmark/augmentations/person_name_perturbation.py)

prob (`float`) –

Probability of substituting a word in the source class with a word in the target class given that a substitution is available.

source_class (`Dict[str, str]`) –

The properties of the source class. The keys of the dictionary should correspond to categories ("gender" only for now) and the values should be the corresponding values. If more than one category is provided. Case-insensitive.

target_class (`Dict[str, str]`) –

Same as source_class, but specifies the target_class.

preserve_gender (`bool`, default:`True`) –

If set to True, we preserve the gender when mapping names of one category to those of another. If we can't find the gender association for a source_word, we randomly pick from one of the target names.

| Parameters: |
| --- |

### ChineseSynonymPerturbation(prob: float, trial_num: int = 10)

Chinese synonyms. For implementation details, see [https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/chinese_antonym_synonym_substitution](https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/chinese_antonym_synonym_substitution)

This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected words excluding punctuations and stopwords.

Perturbation example:

Input: 裸婚，这里的“裸”，指物质财富匮乏的情况下结婚，例如：无房无车无存款，有时候用于强调现实的无奈，也有时候用于强调人对情感的关注。

Output: 裸婚，这里底“裸”，指物质财富匮乏的情况下结婚，譬如说：无房无车无储蓄，有时候用于强调现实的无奈，亦有时候用来强调人士对情感的关注。

### ChineseTyposPerturbation(prob: float, rare_char_prob: float = 0.05, consider_tone: bool = False, word_level_perturb: bool = True)

Chinese typos. For implementation details, see [https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/chinese_butter_fingers_perturbation](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/nlaugmenter/transformations/chinese_butter_fingers_perturbation)

This perturbation adds noise to a text source by randomly replacing Chinese characters or words by other characters or words that share a similar Pinyin.

Perturbation example:

Input: 我想买一部新手机。

Output: 我想买一部新收集。

### MandarinToCantonesePerturbation()

Individual fairness perturbation for Mandarin to Cantonese translation. The implementation is inspired by [https://justyy.com/tools/chinese-converter/](https://justyy.com/tools/chinese-converter/)

Note that this is a rule-based translation system and there are limitations.

### SimplifiedToTraditionalPerturbation()

Individual fairness perturbation for Chinese simplified to Chinese traditional.

## contraction_expansion_perturbation

### ContractionPerturbation()

Contractions. Replaces each expansion with its contracted version.

Perturbation example:

Input: She is a doctor, and I am a student

Output: She's a doctor, and I'm a student

### ExpansionPerturbation()

Expansions. Replaces each contraction with its expanded version.

Perturbation example:

Input: She's a doctor, and I'm a student

Output: She is a doctor, and I am a student

## contrast_sets_perturbation

### ContrastSetsPerturbation()

Contrast Sets are from this paper (currently supported for the BoolQ and IMDB scenarios): [https://arxiv.org/abs/2004.02709](https://arxiv.org/abs/2004.02709)

Original repository can be found at: [https://github.com/allenai/contrast-sets](https://github.com/allenai/contrast-sets)

An example instance of a perturbation for the BoolQ dataset:

```
The Fate of the Furious premiered in Berlin on April 4, 2017, and was theatrically released in the United States on
April 14, 2017, playing in 3D, IMAX 3D and 4DX internationally. . . A spinoff film starring Johnson and Statham’s
characters is scheduled for release in August 2019, while the ninth and tenth films are scheduled for releases on
the years 2020 and 2021.
question: is “Fate and the Furious” the last movie?
answer: no

perturbed question: is “Fate and the Furious” the first of multiple movies?
perturbed answer: yes
perturbation strategy: adjective change.

```

An example instance of a perturbation for the IMDB dataset (from the original paper):

```
Original instance: Hardly one to be faulted for his ambition or his vision, it is genuinely unexpected, then, to see
all Park’s effort add up to so very little. . . .  The premise is promising, gags are copious and offbeat humour
abounds but it all fails miserably to create any meaningful connection with the audience.
Sentiment: negative

Perturbed instance: Hardly one to be faulted for his ambition or his vision, here we
see all Park’s effort come to fruition. . . . The premise is perfect, gags are
hilarious and offbeat humour abounds, and it creates a deep connection with the
audience.
Sentiment: positive

```

## dialect_perturbation

### DialectPerturbation(prob: float, source_class: str, target_class: str, mapping_file_path: Optional[str] = None)

Individual fairness perturbation for dialect.

If mapping_file_path is not provided, (source_class, target_class) should be ("SAE", "AAVE").

prob (`float`) –

Probability of substituting a word in the original class with a word in the target class given that a substitution is available.

source_class (`str`) –

The source dialect that will be substituted with the target dialect. Case-insensitive.

target_class (`str`) –

The target dialect.

mapping_file_path (`Optional[str]`, default:`None`) –

The absolute path to a file containing the word mappings from the source dialect to the target dialect in a json format. The json dictionary must be of type Dict[str, List[str]]. Otherwise, the default dictionary in self.MAPPING_DICTS for the provided source and target classes will be used, if available.

| Parameters: |
| --- |

## extra_space_perturbation

### ExtraSpacePerturbation(num_spaces: int)

A toy perturbation that replaces existing spaces in the text with`num_spaces` number of spaces.

## filler_words_perturbation

### FillerWordsPerturbation(insert_prob = 0.333, max_num_insert = None, speaker_ph = True, uncertain_ph = True, fill_ph = True)

Randomly inserts filler words and phrases in the sentence. Perturbation example:

Input: The quick brown fox jumps over the lazy dog.

Output: The quick brown fox jumps over probably the lazy dog.

## gender_perturbation

### GenderPerturbation(mode: str, prob: float, source_class: str, target_class: str, mapping_file_path: Optional[str] = None, mapping_file_genders: Optional[List[str]] = None, bidirectional: bool = False)

Individual fairness perturbation for gender terms and pronouns.

mode (`str`) –

The mode of the gender perturbation, must be one of "terms" or "pronouns".

prob (`float`) –

Probability of substituting a word in the source class with a word in the target class given that a substitution is available.

source_class (`str`) –

The source gender that will be substituted with the target gender. If mapping_file_path is provided, the source class must be one of the genders in it. If not, it must be exactly one of`male`,`female`, and `neutral. Case-insensitive.

target_class (`str`) –

Same as the source class, but for the target gender.

mapping_file_path (`Optional[str]`, default:`None`) –

The absolute path to a file containing the word mappings from the source gender to the target gender in a json format. The json dictionary must be of type List[List[str, ...]]. It is assumed that 0th index of the inner lists correspond to the 0th gender, 1st index to 1st gender and so on. All word cases are lowered. If mapping_file_path is None, the default dictionary in self.MODE_TO_MAPPINGS for the provided source and target classes will be used, if available.

mapping_file_genders (`Optional[List[str]]`, default:`None`) –

The genders in the mapping supplied in the mapping_file_path. The inner lists read from mapping_file_path should have the same length as the mapping_file_genders. The order of the genders is assumed to reflect the order in the mapping_file_path. Must not be None if mapping_file_path is set. All word cases are lowered.

bidirectional (`bool`, default:`False`) –

Whether we should apply the perturbation in both directions. If we need to perturb a word, we first check if it is in list of source_class words, and replace it with the corresponding target_class word if so. If the word isn't in the source_class words, we check if it is in the target_class words, and replace it with the corresponding source_class word if so.

| Parameters: |
| --- |

## lowercase_perturbation

### LowerCasePerturbation

Simple perturbation turning input and references into lowercase.

## mild_mix_perturbation

### MildMixPerturbation()

Canonical robustness perturbation that composes several perturbations. These perturbations are chosen to be reasonable.

#### contraction_perturbation = ContractionPerturbation()instance-attribute

#### lowercase_perturbation = LowerCasePerturbation()instance-attribute

#### misspelling_perturbation = MisspellingPerturbation(prob=0.1)instance-attribute

#### space_perturbation = SpacePerturbation(max_spaces=3)instance-attribute

## misspelling_perturbation

### MisspellingPerturbation(prob: float)

Replaces words randomly with common misspellings, from a list of common misspellings.

Perturbation example:

Input: Already, the new product is not available.

Output: Aready, the new product is not availible.

prob (`float`) –

probability between [0,1] of perturbing a word to a common misspelling (if we have a common misspelling for the word)

| Parameters: |
| --- |

## person_name_perturbation

### PersonNamePerturbation(prob: float, source_class: Dict[str, str], target_class: Dict[str, str], name_file_path: Optional[str] = None, person_name_type: str = FIRST_NAME, preserve_gender: bool = True)

Individual fairness perturbation for person names.

If name_file_path isn't provided, we use our default name mapping file, which can be found at:

```
https://storage.googleapis.com/crfm-helm-public/source_datasets/augmentations/person_name_perturbation/person_names.txt

```

The available categories in our default file and their values are as follows:

```
If person_name_type == "last_name":

    (1) "race"   => "asian", "chinese", "hispanic", "russian", "white"

If person_name_type == "first_name":

    (1) "race"   => "white_american", "black_american"
    (2) "gender" => "female", "male"

```

The first names in our default file come from Caliskan et al. (2017), which derives its list from Greenwald (1998). The former removed some names from the latter because the corresponding tokens infrequently occurred in Common Crawl, which was used as the training corpus for GloVe. We include the full list from the latter in our default file.

The last names in our default file and their associated categories come from Garg et. al. (2017), which derives its list from Chalabi and Flowers (2014).

prob (`float`) –

Probability of substituting a word in the source class with a word in the target class given that a substitution is available.

source_class (`Dict[str, str]`) –

The properties of the source class. The keys of the dictionary should correspond to categories ("race", "gender", "religion, "age", etc.) and the values should be the corresponding values. If more than one category is provided, the source_names list will be constructed by finding the intersection of the names list for the provided categories. Assuming the 'first_name' mode is selected, an example dictionary can be: {'race': 'white_american'}. Case-insensitive.

target_class (`Dict[str, str]`) –

Same as source_class, but specifies the target_class.

name_file_path (`Optional[str]`, default:`None`) –

The absolute path to a file containing the category associations of names. Each row of the file must have the following format:

```
<name>,<name_type>[,<category>,<value>]*

```

Here is a breakdown of the fields: : The name (e.g. Alex). : Must be one of "first_name" or "last_name". : The name of the category (e.g. race, gender, age, religion, etc.) : Value of the preceding category.

[,,]* denotes that any number of category and value pairs can be appended to a line.

Here are some example lines: li,last_name,race,chinese aiesha,first_name,race,black_american,gender,female

Notes: (1) For each field, the leading and trailing spaces are ignored, but those in between words in a field are kept. (2) All the fields are lowered. (3) It is possible for a name to have multiple associations (e.g. with more than one age, gender etc.)

We use the default file if None is provided.

person_name_type (`str`, default:`FIRST_NAME`) –

One of "first_name" or "last_name". If "last_name", preserve_gender field must be False. Case-insensitive.

preserve_gender (`bool`, default:`True`) –

If set to True, we preserve the gender when mapping names of one category to those of another. If we can't find the gender association for a source_word, we randomly pick from one of the target names.

| Parameters: |
| --- |

## space_perturbation

### SpacePerturbation(max_spaces: int)

A simple perturbation that replaces existing spaces with 0-max_spaces spaces (thus potentially merging words).

## suffix_perturbation

### SuffixPerturbation(suffix: str)

Appends a suffix to the end of the text. Example:

A picture of a dog -> A picture of a dog, picasso

## synonym_perturbation

### SynonymPerturbation(prob: float)

Synonyms. For implementation details, see [https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py](https://github.com/GEM-benchmark/NL-Augmenter/blob/main/nlaugmenter/transformations/synonym_substitution/transformation.py)

This perturbation adds noise to a text source by randomly inserting synonyms of randomly selected words excluding punctuations and stopwords. The space of synonyms depends on WordNet and could be limited. The transformation might introduce non-grammatical segments.

Perturbation example:

Input: This was a good movie, would watch again.

Output: This was a dependable movie, would determine again.

## translate_perturbation

### TranslatePerturbation(language_code: str)

Translates to different languages.

## typos_perturbation

### TyposPerturbation(prob: float)

Typos. For implementation details, see [https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation](https://github.com/GEM-benchmark/NL-Augmenter/tree/main/transformations/butter_fingers_perturbation)

Replaces each random letters with nearby keys on a querty keyboard. We modified the keyboard mapping compared to the NL-augmenter augmentations so that: a) only distance-1 keys are used for replacement, b) the original letter is no longer an option, c) removed special characters (e.g., commas).

Perturbation example:

Input: After their marriage, she started a close collaboration with Karvelas.

Output: Aftrr theif marriage, she started a close collaboration with Karcelas.
