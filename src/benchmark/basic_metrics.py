from typing import List, Callable, Optional, Dict, Tuple
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile

from common.statistic import Stat
from .adapter import AdapterSpec, RequestState
from .metric import Metric
from .metric_service import MetricService
from proxy.tokenizer.auto_token_counter import AutoTokenCounter
from proxy.tokenizer.token_counter import TokenCounter


def check_correctness(problem: Dict, completion: str, timeout: float, completion_id: Optional[int] = None) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil

            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            # Disable functionalities that can make destructive changes to the test.
            reliability_guard()

            # Construct the check program and run it.
            check_program = (
                problem["prompt"] + completion + "\n" + problem["test"] + "\n" + f"check({problem['entry_point']})"
            )

            try:
                exec_globals = {}
                with swallow_io():
                    with time_limit(timeout):
                        # WARNING
                        # This program exists to execute untrusted model-generated code. Although
                        # it is highly unlikely that model-generated code will do something overtly
                        # malicious in response to this test suite, model-generated code may act
                        # destructively due to a lack of model capability or alignment.
                        # Users are strongly encouraged to sandbox this evaluation suite so that it
                        # does not perform destructive actions on their host or network. For more
                        # information on how OpenAI sandboxes its code, see the accompanying paper.
                        # Once you have read this disclaimer and taken appropriate precautions,
                        # uncomment the following line and proceed at your own risk:
                        # exec(check_program, exec_globals)
                        pass
                result.append("passed")
            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir

    manager = multiprocessing.Manager()
    result: List[str] = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return dict(
        task_id=problem["task_id"], passed=result[0] == "passed", result=result[0], completion_id=completion_id,
    )


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """ StringIO that throws an exception when it's read from """

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """ Returns True if the IO object can be read. """
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the 
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if not platform.uname().system == "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins

    builtins.exit = None
    builtins.quit = None

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.fork = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    import shutil

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None

    import sys

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None


def exact_match(gold: Tuple[str, Dict], pred: str) -> float:
    return 1 if gold[0] == pred else 0


def get_num_bytes(text: str) -> int:
    """Compute the byte length of the input string"""
    return len(bytes(text, encoding="utf-8"))


def iou_set_match(gold: Tuple[str, Dict], pred: str) -> float:
    """Compute the intersection over union of the gold and pred sets"""
    pred = pred.split("\n")[0]
    gold_text = gold[0]
    if gold_text == "Nothing.":
        return float(pred == "Nothing.")
    pred = pred.replace(".", "")
    gold_text = gold_text.replace(".", "")
    gold_set = set(gold_text.split(" is ")[-1].split(" and "))
    pred_set = set(pred.split(" is ")[-1].split(" and "))
    return len(gold_set.intersection(pred_set)) / len(gold_set.union(pred_set))


def exact_set_match(gold: Tuple[str, Dict], pred: str) -> float:
    """Compute whether the sets generated exactly match"""
    pred = pred.split("\n")[0]
    gold_text = gold[0]
    if gold_text == "Nothing.":
        return float(pred == "Nothing.")
    pred = pred.replace(".", "")
    gold_text = gold_text.replace(".", "")
    gold_set = set(gold_text.split(" is ")[-1].split(" and "))
    pred_set = set(pred.split(" is ")[-1].split(" and "))
    return float(gold_set == pred_set)


def code_eval(gold: Tuple[str, Dict], pred: str) -> float:
    """Evaluate Code Correctness on test examples"""
    ###### Warning: will execute machine generated code; need to sandbox before executing thisgithub
    return check_correctness(gold[1], pred, 3.0)["passed"]


class BasicMetric(Metric):
    """
    Defines basic metrics which don't require domain knowledge.  This should be
    fairly comprehensive already and we should try to use this as much as possible.
    If we need a different variant, try to generalize this or factor things out.
    It's possible we don't need to subclass this.
    `names` is a list of optional metrics to be specified by the user. Currently only `exact_match` is supported.
    """

    def __init__(self, names: List[str]):
        self.names = names
        self.token_counter: TokenCounter = AutoTokenCounter()

    def compute_reference_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """
        Setup:
        - Gold (correct references): G1 ... Gm
        - Predictions (completions): P1 ... Pk

        For each pair (G, P), we can define a ${score} (e.g., exact match, F1, BLEU).

        We define the following stats:
        - ${score}: max_i score(Gi, P1)
        - ${score}@k: max_{i,j} score(Gi, Pj)
        """

        def compute_metrics_helper(
            name: str,
            score_func: Callable[[Tuple[str, Dict], str], float],
            golds: List[Tuple[str, Dict]],
            preds: List[str],
        ) -> List[Stat]:
            score_1 = max(score_func(gold, preds[0]) for gold in golds)
            score_k = max(score_func(gold, pred) for gold in golds for pred in preds)

            return [
                Stat(name).add(score_1),
                Stat(f"{name}@{adapter_spec.num_outputs}").add(score_k),
            ]

        # maps each string metric name to its associated function
        metric_fn_mapping = {
            "exact_match": exact_match,
            "exact_set_match": exact_set_match,
            "iou_set_match": iou_set_match,
            "code_eval": code_eval,
        }

        reference_metrics = []
        for metric_name in self.names:
            if metric_name in metric_fn_mapping:
                # Gold outputs
                golds = [
                    (reference.output, reference.data)
                    for reference in request_state.instance.references
                    if reference.is_correct
                ]
                assert len(golds) > 0

                # Predicted outputs
                assert request_state.result is not None
                # TODO: Sort the predictions, or take them from the top tokens of the first completion
                #       https://github.com/stanford-crfm/benchmarking/issues/42
                preds = [completion.text.strip() for completion in request_state.result.completions]

                # Apply mapping if exists (e.g., for multiple-choice questions A -> Boston, B -> New York)
                if request_state.output_mapping is not None:
                    preds = [request_state.output_mapping.get(pred) for pred in preds]
                reference_metrics.extend(
                    compute_metrics_helper(metric_name, metric_fn_mapping[metric_name], golds, preds)
                )
            else:
                raise NameError(f"{metric_name} is not in the list of metric functions.")
        return reference_metrics

    def compute_runtime_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute per-token normalized runtime"""
        assert request_state.result is not None

        runtime: float = request_state.result.request_time

        # Compute total number of tokens across completions
        num_tokens: int = sum([len(sequence.tokens) for sequence in request_state.result.completions])
        # Account for the tokens in prompt as well if echo_prompt is False
        if not request_state.request.echo_prompt:
            num_tokens_in_prompt: int = self.token_counter.tokenize_and_count(
                model=request_state.request.model, text=request_state.request.prompt
            )
            num_tokens += num_tokens_in_prompt

        return [Stat("runtime").add(runtime), Stat("normalized_runtime").add(runtime / num_tokens)]

    def compute_language_modeling_metrics(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the logprob and normalization factors for the first completion"""
        assert request_state.result is not None
        sequence = request_state.result.completions[0]
        logprob, num_tokens, num_bytes = sequence.logprob, len(sequence.tokens), get_num_bytes(sequence.text)

        # Ignore the conditioning prefix
        conditioning_prefix_length = 0
        conditioning_prefix_tokens = []
        for token in sequence.tokens:
            if conditioning_prefix_length >= len(adapter_spec.conditioning_prefix):
                break
            conditioning_prefix_tokens.append(token)
            conditioning_prefix_length += len(token.text)
        assert "".join([token.text for token in conditioning_prefix_tokens]) == adapter_spec.conditioning_prefix

        logprob -= sum(token.logprob for token in conditioning_prefix_tokens)
        num_tokens -= len(conditioning_prefix_tokens)
        num_bytes -= get_num_bytes(adapter_spec.conditioning_prefix)

        return [Stat("logprob").add(logprob), Stat("num_tokens").add(num_tokens), Stat("num_bytes").add(num_bytes)]

    def evaluate_generation(
        self, adapter_spec: AdapterSpec, request_state: RequestState, metric_service: MetricService
    ) -> List[Stat]:
        """Compute the reference metrics and language modeling metrics"""
        metrics = []
        if len(request_state.instance.references) > 0:
            metrics.extend(self.compute_reference_metrics(adapter_spec, request_state, metric_service))

        metrics.extend(self.compute_language_modeling_metrics(adapter_spec, request_state, metric_service))
        metrics.extend(self.compute_runtime_metrics(adapter_spec, request_state, metric_service))

        # Future: add F1, BLEU, etc.
        # TODO: pass in arguments to `BasicMetric`
        #       https://github.com/stanford-crfm/benchmarking/issues/44

        return metrics

    def evaluate_references(
        self, adapter_spec: AdapterSpec, reference_request_states: List[RequestState], metric_service: MetricService
    ) -> List[Stat]:
        """
        Setup: for each reference, we have a model score (log probability) and whether it's correct.
        We define the following metrics:
        - correct_rank: if we sort references by their logprobs, what is the ranking of the first correct reference.
        """
        # TODO: https://github.com/stanford-crfm/benchmarking/issues/45
        return []
