from dataclasses import dataclass

from .perturbation import Perturbation
from .perturbation_description import PerturbationDescription


# The implementation below is based on the following list of common misspellings:
# https://en.wikipedia.org/wiki/Wikipedia:Lists_of_common_misspellings/For_machines
@dataclass
class MisspellingPerturbation(Perturbation):
    """
    Replaces words randomly with common misspellings, from a list of common misspellings.
    Perturbation example:
    Input:
        Already, the new product is not available.
    Output:
        Aready, the new product is not availible.
    """

    @dataclass(frozen=True)
    class Description(PerturbationDescription):
        name: str

    name: str = "misspellings"

    def __init__(self, prob: float):
        '''
        prob (float): probability between [0,1] of perturbing a word to a common misspelling (if we have a common misspelling for the word)
        '''
        self.prob = prob
        misspellings_file = Path(__file__).resolve().expanduser().parent / 'correct_to_misspelling.json'
        with open(misspellings_file, 'r') as f:
                self.correct_to_misspelling = json.load(f)
        self.detokenizer = TreebankWordDetokenizer()

    @property
    def description(self) -> PerturbationDescription:
        return MisspellingPerturbation.Description(self.name)

    def count_upper(self, word: str) -> int:
        return sum(c.isupper() for c in word)

    def perturb_word(self, word: str) -> str:
        perturbed = False
        if word in self.correct_to_misspelling and np.random.rand() < self.prob:
            word = str(np.random.choice(self.correct_to_misspelling[word]))
            perturbed = True
        return word, perturbed

    def perturb_seed(self, text: str, seed: int=0) -> str:
        words = word_tokenize(text)
        new_words = []
        for word in words:
            # check if word in original form is in dictionary
            word, perturbed = self.perturb_word(word)
            if not perturbed and self.count_upper(word) == 1 and word[0].isupper():
                # if only the first letter is upper case
                # check if the lowercase version is in dict
                lowercase_word = word.lower()
                word, perturbed = self.perturb_word(lowercase_word)
                word = word[0].upper() + word[1:]
            new_words.append(word)
        perturbed = self.detokenizer.detokenize(new_words)
        return perturbed

    def perturb(self, text: str) -> str:
        return self.perturb_seed(text)
