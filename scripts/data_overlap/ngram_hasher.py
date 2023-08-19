from typing import List, Tuple

RABIN_KARP_PRIME = 172001 # chosen as a prime larger than 171,476, estimate vocab size of english (these are the base of exponent)
CHAR_PRIME = 401 # chosen as a prime larger than 256, UTF-8 space, and distant from 2^n; 
class RabinKarpHash:
    """
    (ignoring mods here for simplicity)
    Initial hash:
        for a List[int] i_0...i_n with a window w s.t. w <= n
        hash is i_0 * 256 ^ {w-1} + i_1 * 256 ^ {w-2} + ... + i_{w-1}

    high_coefficient:
        256 * {w-1}
    
    Is it 256 because 2^8 = 256, and the expected input is UTF-8 chars?

    In our case, it probably makes more sense to use a large prime p

    and convert words/tokens into ints (probably by multiplying chars by a prime or 256)

    I'll use 196613 as it's estimated there are 171,476 words in current use
    in English language, and it's prime
    """
    def __init__(self, int_sequence: List[int], window_size: int, mod: int = 8554560727166512717):
        self.window_start = 0
        self.window_end = window_size
        self.mod = mod
        self.high_coefficient = pow(RABIN_KARP_PRIME, window_size - 1, mod)
        self.current_hash = 0
        self.int_sequence = int_sequence

        """
        Is window_size the number of n-grams (i.e., tokens) or is it the number of characters? The comment says:

        # for i in range(window_size):
        #     self.hash = (self.hash * 256 + ord(self.text[i])) % mod

        ord only acts at a character level, but the logic only makes sense for n-grams/tokens (otherwise we have no way to configure N, the length of n-grams)
        """
        for i in range(window_size):
            self.current_hash = (self.current_hash * RABIN_KARP_PRIME + int_sequence[i]) % mod
        if self.current_hash < 0:
            self.current_hash += mod

    def mult_mod(self, a: int, b: int, k: int) -> int:
        return ((a * b) % k)

    """
    Initial hash:
        for a List[int] i_0...i_n with a window w s.t. w <= n
        hash is i_0 * 256 ^ {w-1} + i_1 * 256 ^ {w-2} + ... + i_{w-1}
    
    start_piece = i_0 * 256 ^ {w-1}

    subtract the start piece and multiply by 256, then add the current value, so we have
        hash is i_1 * 256 ^ {w-1} + i_2 * 256 ^ {w-2} + ... + i_{w}

    """
    def update(self) -> None:
        start_piece = self.mult_mod(self.int_sequence[self.window_start], self.high_coefficient, self.mod)
        self.current_hash = self.mult_mod(self.current_hash - start_piece, RABIN_KARP_PRIME, self.mod)
        self.current_hash = (self.current_hash + self.int_sequence[self.window_end]) % self.mod
        if self.current_hash < 0:
            self.current_hash += self.mod
        self.window_start += 1
        self.window_end += 1

def compute_hashes(int_sequence: List[int], window_size: int) -> List[Tuple[int, int, int]]:
    """
    Previously was 
    rk_hash = RabinKarpHash(
        int_sequence,
        min(window_size, len(int_sequence))
    )
    but we don't want anything to happen when len(int_sequence) < window_size
    """
    if len(int_sequence) < window_size:
        return []
    rk_hash = RabinKarpHash(
        int_sequence,
        window_size
    )
    hashes = [(rk_hash.current_hash, rk_hash.window_start, rk_hash.window_end)]
    for i in range(len(int_sequence) - window_size):
        rk_hash.update()
        hash_info = (rk_hash.current_hash, rk_hash.window_start, rk_hash.window_end)
        hashes.append(hash_info)

    return hashes

# mod chosen as large prime of similar size to rabin karp, but distinct
def hash_token(token: str, mod: int = 8554560727166512181):
    hash_value = 0
    for ch in token:
        hash_value = (hash_value * CHAR_PRIME + ord(ch)) % mod
    return hash_value

def get_ngram_hashes(tokens: List[str], n: int):
    hashed_tokens = [hash_token(token) for token in tokens]
    return compute_hashes(hashed_tokens, n)
