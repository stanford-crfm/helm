# Understanding the code

This section will present a high-level overview of the codebase. For a more detailed explanation of the code, refer to the [Developer Guide](developer_guide.md) that explains more advanced concepts.

## Request

If you want to get a completion, you will need to send a `Request` to a `Client`. You can check out the implementation [here](https://github.com/stanford-crfm/helm/blob/1dbd0a7d8edc92a59440f6c0bda7a0aaf30c8676/src/helm/common/request.py#L8-L73). A `Request` is a dataclass that contains the following fields:

- `model (str)`: The name of the model to send the request to.
- `prompt (str)`: The prompt to send to the model.
- `embedding (bool)`: Whether to query embedding instead of text response..
- `max_tokens (int)`: The maximum number of tokens to generate.
- `temperature (float)`: The temperature to use for sampling.
- `num_completions (int)`: The number of completions to generate.
- `top_k_per_token (int)`: ake this many highest probability candidates per token in the completion.
- `top_p (int)`: An alternative to sampling with temperature, called nucleus sampling, where the cumulative probability of tokens to sample from is constrained.
- `stop_sequences (List[str])`: A list of sequences to stop generation at.
- `echo_prompt (bool)`: Whether to include the prompt as a prefix to the completion.
- `random (Optional[str])`: A random seed to use for sampling. This is useful if you want different completions for the same prompt. Each entry is a `dict` that should contain a `"role"` and a `"content"`. If specified, this will override the `prompt` field.
- `messages (Optional[List[Dict[str, str]]])`: A list of messages for chat models.
- `presence_penalty (float)`: Penalize repetition in the generated text. *(only supported by OpenAI and Writer)*.
- `frequency_penalty (float)`: Also penalize repetitions. *(only supported by OpenAI and Writer)*.

If you have a `client` you can simply send a `Request` to get a `RequestResult`:

```python
from helm.common.request import Request

request = Request(
    model="openai/curie",
    prompt="Hello, my name is",
    max_tokens=10,
)
result = client.make_request(request)
```

Let's now see what contains a `RequestResult in the next section.


## RequestResult

A `RequestResult` is a dataclass that contains the response from a client after making a request. Youc an check out our implementation [here](https://github.com/stanford-crfm/helm/blob/1dbd0a7d8edc92a59440f6c0bda7a0aaf30c8676/src/helm/common/request.py#L149-L202). Here are the fields of a `RequestResult`:

- `success (bool)`: Whether the request was successful. If not, the field `error` will contain the error message.
- `error (Optional[str])`: The error message if the request was not successful.
- `completions (list[Sequence])`: The list of completions. Each completion is a `Sequence` object (code [here](https://github.com/stanford-crfm/helm/blob/1dbd0a7d8edc92a59440f6c0bda7a0aaf30c8676/src/helm/common/request.py#L103-L133)). A quick high-level description is that a `Sequence` contains a `text (str)` field, a `tokens (List[Token])` field that contains th elist of tokens (text of the token and logprob) and a `logprob (float)` field that contains the logprob of the sequence.
- `embedding (List[float])`: Fixed dimensional embedding corresponding to the entire prompt.
- `cached (bool)`: Whether the response was cached.
- `error_flags (Optional[ErrorFlags])`: The flags define how to handle the error. If `error_flags.is_retriable` is True, then the request will be retried 7 more times. If `error_flags.is_fatal` is True, then the request will cause the run to fail, otherwise the completion would be simply replaced by an empty string for the metrics.
- `batch_size (Optional[int])`: The batch size used for the request. *(TogetherClient only)*.
- `batch_request_time (Optional[float])`: The time it took to process the batch. *(TogetherClient only)*.
- `request_time (Optional[float])`: The time it took to make the request.
` -request_datetime (Optional[datetime])`: The date and time the request was made.

If you want to get check that the request was successful and if so, get the completions, you can do the following:

```python
if not result.success:
    # By default, if not specified, an error is considered fatal.
    if not result.error_flags or result.error_flags.is_fatal:
        raise Exception(f"Request failed with error: {result.error}")
    else:
        print(f"Request failed with error: {result.error}")
else:
    completions = result.completions
    print("The request was successful. Here are the completions:")
    for completion in completions:
        print(f" - {completion.text}")
```