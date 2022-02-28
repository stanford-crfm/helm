# flake8: noqa
# type: ignore

"""Core function for APPS evaluation.

Adapted from
    https://github.com/hendrycks/apps/blob/main/eval/test_one_solution.py

The original file is very messy and would trigger type-check errors.
We want to only minimally modify it to ensure its functionality isn't modified.
Split this out so we can disable type checking for the entire file.
"""

from enum import Enum
import faulthandler
from io import StringIO
import json
import os
import signal
import sys
from typing import List, Union
from unittest.mock import patch, mock_open

import numpy as np
from pyext import RuntimeModule

from common.hierarchical_logger import hlog


class CodeType(Enum):
    call_based = 0
    standard_input = 1


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


signal.signal(signal.SIGALRM, timeout_handler)
timeout = 4  # seconds


# used to capture stdout as a list
# from https://stackoverflow.com/a/16571630/6416660
# alternative use redirect_stdout() from contextlib
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        # Make closing the StringIO a no-op
        self._stringio.close = lambda x: 1
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


def run_test(root: str, test: str) -> List[Union[int, bool]]:
    """Run a single test.

    Args:
        root: Path to the problem folder.
        test: Candidate source code solution in string format.
    Returns:
        A list of numbers/booleans, one for each test case.
        -2 means compile error, -1 means runtime error, `True` means passed, `False` means failed.
    """

    input_output_fname = os.path.join(root, "input_output.json")
    if os.path.exists(input_output_fname):
        with open(input_output_fname) as f:
            in_outs = json.load(f)
            if in_outs.get("fn_name") is None:
                which_type = CodeType.standard_input  # Standard input
                method_name = None
            else:
                which_type = CodeType.call_based  # Call-based
                method_name = in_outs["fn_name"]

    results = []
    sol = (
        "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, "
        "combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, "
        "ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, "
        "fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as "
        "np\nimport random\nimport heapq\nfrom heapq import *\n"
    )

    if which_type == CodeType.call_based:
        sol += test
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            if "class Solution" not in test:
                tmp = tmp_sol
            else:
                tmp = tmp_sol.Solution()
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            hlog(f"type 0 compilation error = {e}")
            results.append(-2)
            return results
        signal.alarm(0)

    elif which_type == CodeType.standard_input:
        tmp_test = test.split("\n")

        new_test = []
        for x in tmp_test:
            if (not x.startswith("from ")) and (not x.startswith("import ")):
                new_test.append("\t" + x + "\n")
            else:
                new_test.append(x + "\n")
        tmp_test = new_test

        new_test = ""
        started = False
        for i in tmp_test:
            if i.startswith("\t") and not started:
                new_test += "stdin = sys.stdin\nstdout = sys.stdout\n"
                new_test += "def code():\n"
                new_test += i
                started = True
            elif started and ((i.startswith("from ")) or (i.startswith("import "))):
                new_test += "\t" + i
            else:
                new_test += i
        tmp_test = new_test

        sol += tmp_test
        method_name = "code"
        signal.alarm(timeout)
        try:
            tmp_sol = RuntimeModule.from_string("tmp_sol", "", sol)
            tmp = tmp_sol
            signal.alarm(0)
        except Exception as e:
            signal.alarm(0)
            hlog(f"type 1 compilation error = {e}")
            results.append(-2)
            return results
        signal.alarm(0)
    try:
        method = getattr(tmp, method_name)  # get_attr second arg must be str
    except:
        signal.alarm(0)
        e = sys.exc_info()
        hlog(f"unable to get function error = {e}")
        return results

    for index, inputs in enumerate(in_outs["inputs"]):
        # JSON forces dictionaries to have string keys; this undoes this (assuming a singleton list)
        try:
            if isinstance(inputs[0], dict):
                inputs = [{int(k): v for k, v in inputs[0].items()}]
        except:
            pass

        try:
            if isinstance(in_outs["outputs"][index], dict):
                in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index].items()}]
        except:
            pass

        try:
            if isinstance(in_outs["outputs"][index][0], dict):
                in_outs["outputs"][index] = [{int(k): v for k, v in in_outs["outputs"][index][0].items()}]
        except:
            pass

        if which_type == CodeType.call_based:  # Call-based
            signal.alarm(timeout)
            faulthandler.enable()
            try:
                output = method(*inputs)

                # ground truth sequences are not tuples
                if isinstance(output, tuple):
                    output = list(output)

                tmp_result = output == in_outs["outputs"][index]
                if isinstance(in_outs["outputs"][index], list) and in_outs["outputs"][index]:
                    tmp_result = tmp_result or (output == in_outs["outputs"][index][0])

                # ground truth sequences are not tuples
                try:
                    if isinstance(output[0], tuple):
                        tmp_result = tmp_result or ([list(x) for x in output] == in_outs["outputs"][index][0])
                except:
                    pass
                results.append(tmp_result)

                # reset the alarm
                signal.alarm(0)
            except Exception as e:
                signal.alarm(0)
                faulthandler.disable()
                hlog(f"Standard input runtime error or time limit exceeded error = {e}")
                results.append(-1)
                continue
            faulthandler.disable()
            signal.alarm(0)
        elif which_type == CodeType.standard_input:  # Standard input
            faulthandler.enable()
            signal.alarm(timeout)
            passed = False

            if isinstance(inputs, list):
                inputs = "\n".join(inputs)
            if isinstance(in_outs["outputs"][index], list):
                in_outs["outputs"][index] = "\n".join(in_outs["outputs"][index])

            with Capturing() as output:
                try:
                    call_method(method, inputs)
                    # reset the alarm
                    signal.alarm(0)
                    passed = True
                except Exception as e:
                    # runtime error or took too long
                    signal.alarm(0)
                    hlog(f"Call-based runtime error or time limit exceeded error = {repr(e)}{e}")
                    results.append(-1)
                signal.alarm(0)

            if not passed:
                continue

            if custom_compare_(output, in_outs["outputs"][index]):
                tmp_result = True
                results.append(tmp_result)
                continue

            # ground truth sequences are expressed as lists not tuples
            if isinstance(output, tuple):
                output = list(output)

            tmp_result = False
            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
                    if isinstance(output[0], str):
                        tmp_result = tmp_result or ([e.strip() for e in output] == in_outs["outputs"][index])
            except Exception as e:
                hlog(f"Failed check1 exception = {e}")
                pass

            if isinstance(tmp_result, bool) and tmp_result:
                results.append(tmp_result)
                continue

            # try one more time without \n
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = i.split("\n")
                    in_outs["outputs"][index][tmp_index] = [
                        x.strip() for x in in_outs["outputs"][index][tmp_index] if x
                    ]
            else:
                in_outs["outputs"][index] = in_outs["outputs"][index].split("\n")
                in_outs["outputs"][index] = list(filter(len, in_outs["outputs"][index]))
                in_outs["outputs"][index] = list(map(lambda x: x.strip(), in_outs["outputs"][index]))

            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
            except Exception as e:
                hlog(f"Failed check2 exception = {e}")
                pass

            if isinstance(tmp_result, bool) and tmp_result:
                results.append(tmp_result)
                continue

            # try by converting the output into a split up list too
            if isinstance(output, list):
                output = list(filter(len, output))

            if isinstance(tmp_result, bool) and tmp_result:
                results.append(tmp_result)
                continue

            try:
                tmp_result = output == [in_outs["outputs"][index]]
                if isinstance(in_outs["outputs"][index], list):
                    tmp_result = tmp_result or (output == in_outs["outputs"][index])
            except Exception as e:
                hlog(f"Failed check3 exception = {e}")
                pass

            try:
                output_float = [float(e) for e in output]
                gt_float = [float(e) for e in in_outs["outputs"][index]]
                tmp_result = tmp_result or (
                    (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                )
            except:
                pass
            try:
                if isinstance(output[0], list):
                    output_float = [float(e) for e in output[0]]
                    gt_float = [float(e) for e in in_outs["outputs"][index][0]]
                    tmp_result = tmp_result or (
                        (len(output_float) == len(gt_float)) and np.allclose(output_float, gt_float)
                    )
            except:
                pass

            if isinstance(tmp_result, bool) and tmp_result:
                results.append(tmp_result)
                continue

            # try by converting the stuff into split up list
            if isinstance(in_outs["outputs"][index], list):
                for tmp_index, i in enumerate(in_outs["outputs"][index]):
                    in_outs["outputs"][index][tmp_index] = set(i.split())
            else:
                in_outs["outputs"][index] = set(in_outs["outputs"][index].split())

            try:
                tmp_result = output == in_outs["outputs"][index]
            except Exception as e:
                hlog(f"Failed check4 exception = {e}")
                continue

            if isinstance(tmp_result, bool) and tmp_result:
                results.append(tmp_result)
                continue

                # try by converting the output into a split up list too
            if isinstance(output, list):
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = i.split()
                output = list(filter(len, output))
                for tmp_index, i in enumerate(output):
                    output[tmp_index] = set(i)
            else:
                output = output.split()
                output = list(filter(len, output))
                output = set(output)

            try:
                tmp_result = set(frozenset(s) for s in output) == set(frozenset(s) for s in in_outs["outputs"][index])
            except Exception as e:
                hlog(f"Failed check5 exception = {e}")

            # if they are all numbers, round so that similar numbers are treated as identical
            try:
                tmp_result = tmp_result or (
                    set(frozenset(round(float(t), 3) for t in s) for s in output)
                    == set(frozenset(round(float(t), 3) for t in s) for s in in_outs["outputs"][index])
                )
            except Exception as e:
                hlog(f"Failed check6 exception = {e}")

            results.append(tmp_result)

    return results


def custom_compare_(output, ground_truth):
    if isinstance(output, list):
        output_1 = "\n".join(output)
        if stripped_string_compare(output_1, ground_truth):
            return True

    if isinstance(output, list):
        output_2 = [o.lstrip().rstrip() for o in output]
        output_2 = "\n".join(output_2)
        if stripped_string_compare(output_2, ground_truth):
            return True

    return False


def stripped_string_compare(s1, s2):
    s1 = s1.lstrip().rstrip()
    s2 = s2.lstrip().rstrip()
    return s1 == s2


def call_method(method, inputs):
    if isinstance(inputs, list):
        inputs = "\n".join(inputs)

    inputs_line_iterator = iter(inputs.split("\n"))

    @patch("builtins.open", mock_open(read_data=inputs))
    @patch("sys.stdin", StringIO(inputs))
    @patch("sys.stdin.readline", lambda *args: next(inputs_line_iterator))
    @patch("sys.stdin.readlines", lambda *args: inputs.split("\n"))
    @patch("sys.stdin.read", lambda *args: inputs)
    def _inner_call_method(_method):
        try:
            return _method()
        except SystemExit as e:
            pass
        finally:
            pass

    return _inner_call_method(method)
