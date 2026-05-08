# flake8: noqa
# fmt: off
# The following code was reproduced from the following URL:
# https://github.com/google-deepmind/eval_hub/blob/67b7fd29b2205ee0a3226e0d3e5d74140a253b42/eval_hub/mrcr_v2/run_evaluation.py

# Copyright 2026 DeepMind Technologies Limited.
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

import difflib


def mrcr_v2_metric(prediction: str, target: str) -> float:
  """Computes the MRCR V2 metric.

  This metric uses difflib SequenceMatcher to compute a notion of approximate
  edit distance between the target reference and the model's output, scaled
  to lie within [0, 1]. 1 is a perfect match, 0 is a non-match. Additionally,
  the metric score is 0 if the random string is not the first 12 characters of
  the output (after stripping whitespace). For outputs where there are multiple
  matches of the random string, we keep only the last one and only consider the
  substring of the output following this last match.

  Args:
    prediction: The model's output.
    target: The target output. Note that it contains the random hash string as a
      prefix.

  Returns:
    The MRCR V2 metric score contained in the interval [0, 1].
  """
  if not isinstance(prediction, str) or not prediction:
    return 0.0

  target = target.strip()
  # The target format is strictly: [12-char-hash][actual-content]
  if len(target) < 12:
    return 0.0

  random_hash = target[:12]
  target_ref = target[12:].strip()
  prediction = prediction.strip()

  # Find the *last* instance of the random hash in the prediction
  start_index = prediction.rfind(random_hash)

  if start_index == -1:
    return 0.0

  # Extract content immediately following the last hash
  # start_index + 12 skips past the hash itself.
  prediction_content = prediction[start_index + 12 :].strip()

  d = difflib.SequenceMatcher(a=target_ref, b=prediction_content)
  return d.ratio()
