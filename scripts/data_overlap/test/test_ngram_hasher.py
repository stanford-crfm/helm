import pytest
from ngram_hasher import RabinKarpHash, compute_hashes, hash_token, get_ngram_hashes


@pytest.fixture
def example_int_sequence():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def example_tokens():
    return ["apple", "banana", "cherry", "date", "elderberry"]


def test_RabinKarpHash_initialization(example_int_sequence):
    window_size = 3
    rk_hash = RabinKarpHash(example_int_sequence, window_size)

    assert isinstance(rk_hash, RabinKarpHash)
    assert rk_hash.current_hash != 0


def test_RabinKarpHash_update(example_int_sequence):
    window_size = 3
    rk_hash = RabinKarpHash(example_int_sequence, window_size)

    initial_hash = rk_hash.current_hash
    rk_hash.update()
    updated_hash = rk_hash.current_hash

    assert updated_hash != initial_hash


def test_compute_hashes(example_int_sequence):
    window_size = 3
    hashes = compute_hashes(example_int_sequence, window_size)

    assert isinstance(hashes, list)
    assert len(hashes) == len(example_int_sequence) - window_size + 1

    for h in hashes:
        assert isinstance(h, tuple)
        assert len(h) == 3

    distinct_hashes = set(hash[0] for hash in hashes)
    assert len(distinct_hashes) == len(hashes)


def test_hash_token():
    token = "apple"
    hash_value = hash_token(token)

    assert isinstance(hash_value, int)
    assert hash_value != 0


def test_get_ngram_hashes(example_tokens):
    n = 2
    hashes = get_ngram_hashes(example_tokens, n)

    assert isinstance(hashes, list)
    assert len(hashes) == len(example_tokens) - n + 1

    for h in hashes:
        assert isinstance(h, tuple)
        assert len(h) == 3

    distinct_hashes = set(hash[0] for hash in hashes)
    assert len(distinct_hashes) == len(hashes)
