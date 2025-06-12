import nltk
from importlib.metadata import version


def install_nltk_resources():
    """Install resources for nltk tokenizers, which is required for bleu and rouge scores."""
    # Install "punkt_tab" for nltk>=3.9.1 or "punkt" for nltk<=3.8.1
    #
    # Note that nltk 3.9.0 is disallowed due to https://github.com/nltk/nltk/issues/3308
    #
    # "punkt" is not longer supported for newer versions of nltk due to a security issue
    # and has been replaced by "punkt_tab". For more information, see:
    #
    # - https://github.com/stanford-crfm/helm/issues/2926
    # - https://github.com/nltk/nltk/issues/3293
    # - https://github.com/nltk/nltk/issues/3266
    # - https://nvd.nist.gov/vuln/detail/CVE-2024-39705
    #
    # TODO: Remove support for nltk<=3.8.1 and only install "punkt_tab"
    nltk_major_version, nltk_minor_version = [int(v) for v in version("nltk").split(".")[0:2]]
    if nltk_major_version < 3:
        raise Exception("ntlk version <3 is not supported")
    if nltk_major_version == 3 and nltk_minor_version <= 8:
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt")
    else:
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab")
