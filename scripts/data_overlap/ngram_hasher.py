from typing import List, Tuple

# A prime number chosen as a base for exponentiation, larger than 171,476 (estimated vocab size of English words)
RABIN_KARP_PRIME = 172001

# A prime number chosen as a base for hashing characters, larger than 256 (UTF-8 character space)
CHAR_PRIME = 401


class RabinKarpHash:
    """
    Rabin-Karp rolling hash for hashing sequences of integers.

    Initial hash:
        For a List[int] i_0...i_n with a window w where w <= n,
        the hash is calculated as:
        i_0 * RABIN_KARP_PRIME^(w-1) + i_1 * RABIN_KARP_PRIME^(w-2) + ... + i_{w-1}

    high_coefficient:
        RABIN_KARP_PRIME^(w-1)
    """

    def __init__(self, int_sequence: List[int], window_size: int, mod: int = 8554560727166512717):
        self.window_start = 0
        self.window_end = window_size
        self.mod = mod
        self.high_coefficient = pow(RABIN_KARP_PRIME, window_size - 1, mod)
        self.current_hash = 0
        self.int_sequence = int_sequence

        for i in range(window_size):
            self.current_hash = (self.current_hash * RABIN_KARP_PRIME + int_sequence[i]) % mod
        if self.current_hash < 0:
            self.current_hash += mod

    def mult_mod(self, a: int, b: int, k: int) -> int:
        return (a * b) % k

    def update(self) -> None:
        """
        Update the rolling hash value when the window slides by one element.

        - To efficiently update the hash:
        1. Subtract the contribution of the element moving out of the window (start_piece).
        2. Multiply the result by RABIN_KARP_PRIME.
        3. Add the contribution of the new element entering the window.

        Then update the window indices
        """
        start_piece = self.mult_mod(self.int_sequence[self.window_start], self.high_coefficient, self.mod)
        self.current_hash = self.mult_mod(self.current_hash - start_piece, RABIN_KARP_PRIME, self.mod)
        self.current_hash = (self.current_hash + self.int_sequence[self.window_end]) % self.mod
        if self.current_hash < 0:
            self.current_hash += self.mod
        self.window_start += 1
        self.window_end += 1


def compute_hashes(int_sequence: List[int], window_size: int) -> List[Tuple[int, int, int]]:
    """
    Computes Rabin-Karp rolling hashes for a list of integers.

    Args:
        int_sequence (List[int]): List of integers.
        window_size (int): Size of the rolling window.

    Returns:
        List[Tuple[int, int, int]]: List of tuples containing hash value, window start, and window end.
    """
    if len(int_sequence) < window_size:
        return []
    rk_hash = RabinKarpHash(int_sequence, window_size)
    hashes = [(rk_hash.current_hash, rk_hash.window_start, rk_hash.window_end)]
    for i in range(len(int_sequence) - window_size):
        rk_hash.update()
        hash_info = (rk_hash.current_hash, rk_hash.window_start, rk_hash.window_end)
        hashes.append(hash_info)

    return hashes


# mod chosen as large prime of similar size to rabin karp, but distinct
def hash_token(token: str, mod: int = 8554560727166512181):
    """
    Hashes a string token into an integer.

    Args:
        token (str): Input token.
        mod (int): Modulus for hashing.

    Returns:
        int: Hash value.
    """
    hash_value = 0
    for ch in token:
        hash_value = (hash_value * CHAR_PRIME + ord(ch)) % mod
    return hash_value


def get_ngram_hashes(tokens: List[str], n: int):
    """
    Computes Rabin-Karp rolling hashes for a list of string tokens.

    Args:
        tokens (List[str]): List of string tokens.
        n (int): Size of the rolling window.

    Returns:
        List[Tuple[int, int, int]]: List of tuples containing hash value, window start, and window end.
    """
    hashed_tokens = [hash_token(token) for token in tokens]
    return compute_hashes(hashed_tokens, n)
