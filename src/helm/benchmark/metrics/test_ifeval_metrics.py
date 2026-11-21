import subprocess
import sys

# Fresh interpreter: the race only happens while langdetect is still loading its profiles.
_SCRIPT = """
from concurrent.futures import ThreadPoolExecutor
from helm.benchmark.metrics.ifeval.instructions import LowercaseLettersEnglishChecker

checker = LowercaseLettersEnglishChecker("change_case:english_lowercase")
checker.build_description()
response = "the quick brown fox jumps over the lazy dog and then runs back into the forest to find its friends."
with ThreadPoolExecutor(max_workers=8) as executor:
    assert all(executor.map(checker.check_following, [response] * 100))
"""


def test_langdetect_in_thread_pool():
    process = subprocess.run([sys.executable, "-c", _SCRIPT], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
