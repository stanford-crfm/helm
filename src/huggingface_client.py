import transformers

# TODO: this class is out of date and needs to be revamped.  It's not clear we
# want this rather than just using Hugging Face's APIs.
# https://huggingface.co/gpt2
class HuggingFaceClient(object):
    def __init__(self):
        self.generators = {}  # model name -> generator

    def complete(self, **kwargs):
        # Get generator
        prompt = kwargs["prompt"]
        model = kwargs["model"]
        max_tokens = kwargs["maxTokens"]
        top_k = kwargs["topK"]
        if model in self.generators:
            generator = self.generators[model]
        else:
            generator = transformers.pipeline("text-generation", model=model)
            self.generators[model] = generator
        max_length = len(prompt.split(" ")) + max_tokens
        response = generator(prompt, max_length=max_length, num_return_sequences=top_k)

        completions = []
        for item in response:
            completions.append(
                {"text": item["generated_text"][len(prompt) :],}
            )
        return {
            "completions": completions,
        }
