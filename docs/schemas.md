---
title: Schemas
---
# Schemas

## Scenariodataclass

A scenario represents a (task, data distribution). It is usually based on some raw dataset and is converted into a list of`Instance` s. Override this class.

Note: the constructor should be lightweight,`get_instances` should do all the heavy lifting.

### name: str = field(init=False)class-attributeinstance-attribute

Short unique identifier of the scenario

### description: str = field(init=False)class-attributeinstance-attribute

Description of the scenario (task, data)

### tags: List[str] = field(init=False)class-attributeinstance-attribute

Extra metadata (e.g., whether this is a question answering or commonsense task)

### definition_path: str = field(init=False)class-attributeinstance-attribute

Where the scenario subclass for`self` is defined.

### __post_init__() -> None

### get_instances(output_path: str) -> List[Instance]abstractmethod

Does the main work in the`Scenario`(e.g., download datasets, convert it into a list of instances).

### render_lines(instances: List[Instance]) -> List[str]

### get_metadata() -> ScenarioMetadata

## ScenarioStatedataclass

A`ScenarioState` represents the output of adaptation. Contains a set of`RequestState` that were created and executed (a`ScenarioState` could be pre-execution or post-execution).

### adapter_spec: AdapterSpecinstance-attribute

### request_states: List[RequestState]instance-attribute

### annotator_specs: Optional[List[AnnotatorSpec]] = Noneclass-attributeinstance-attribute

### __post_init__()

### get_request_states(train_trial_index: int, instance: Instance, reference_index: Optional[int]) -> List[RequestState]

## RequestStatedataclass

A`RequestState` represents a single`Request` made on behalf of an`Instance`. It should have all the information that's needed later for a`Metric` to be able to understand the`Request` and its`RequestResult`.

### instance: Instanceinstance-attribute

Which instance we're evaluating

### reference_index: Optional[int]instance-attribute

Which reference of the instance we're evaluating (if any)

### request_mode: Optional[str]instance-attribute

Which request mode ("original" or "calibration") of the instance we're evaluating (if any) (for ADAPT_MULTIPLE_CHOICE_SEPARATE_CALIBRATED)

### train_trial_index: intinstance-attribute

Which training set this request is for

### output_mapping: Optional[Dict[str, str]]instance-attribute

How to map the completion text back to a real output (e.g., for multiple choice, "B" => "the second choice")

### request: Requestinstance-attribute

The request that is actually made

### result: Optional[RequestResult]instance-attribute

The result of the request (filled in when the request is executed)

### num_train_instances: intinstance-attribute

Number of training instances (i.e., in-context examples)

### prompt_truncated: boolinstance-attribute

Whether the prompt (instructions + test input) is truncated to fit the model's context window.

### num_conditioning_tokens: int = 0class-attributeinstance-attribute

The number of initial tokens that will be ignored when computing language modeling metrics

### annotations: Optional[Dict[str, Any]] = Noneclass-attributeinstance-attribute

Output of some post-processing step that is needed for the metric to understand the request Should match the annotator's name to an Annotation (usually a list of dictionaries for each completion) Example: parsing, rendering an image based on the text completion, etc.

### __post_init__()

### render_lines() -> List[str]

## Instancedataclass

An`Instance` represents one data point that we're evaluating on (e.g., one question in a QA task). Note:`eq=False` means that we hash by the identity.

### input: Inputinstance-attribute

The input

### references: List[Reference]instance-attribute

References that helps us evaluate

### split: Optional[str] = Noneclass-attributeinstance-attribute

Split (e.g., train, valid, test)

### sub_split: Optional[str] = Noneclass-attributeinstance-attribute

Sub split (e.g. toxic, non-toxic)

### id: Optional[str] = Noneclass-attributeinstance-attribute

Used to group Instances that were created from a particular Instance through data augmentation

### perturbation: Optional[PerturbationDescription] = Noneclass-attributeinstance-attribute

Description of the Perturbation that was applied when creating this Instance

### contrast_inputs: Optional[List[Input]] = Noneclass-attributeinstance-attribute

Perturbed input as defined by contrast sets (if available)

### contrast_references: Optional[List[List[Reference]]] = Noneclass-attributeinstance-attribute

References for the perturbed input above (if available)

### extra_data: Optional[Dict[str, Any]] = Noneclass-attributeinstance-attribute

Extra data required by the scenario e.g. chain-of-thought annotations

### first_correct_reference: Optional[Reference]property

Return the first correct reference.

### all_correct_references: List[Reference]property

Return all correct references.

### render_lines() -> List[str]

## Referencedataclass

A`Reference` specifies a possible output and how good/bad it is. This could be used to represent multiple reference outputs which are all acceptable (e.g., in machine translation) or alternatives (e.g., in a multiple-choice exam).

### output: Outputinstance-attribute

The output

### tags: List[str]instance-attribute

Extra metadata (e.g., whether it's correct/factual/toxic)

### is_correct: boolproperty

### render_lines() -> List[str]

## PerturbationDescriptiondataclass

DataClass used to describe a Perturbation

### name: strinstance-attribute

Name of the Perturbation

### robustness: bool = Falseclass-attributeinstance-attribute

Whether a perturbation is relevant to robustness. Will be used to aggregate perturbations metrics

### fairness: bool = Falseclass-attributeinstance-attribute

Whether a perturbation is relevant to fairness. Will be used to aggregate perturbations metrics

### computed_on: str = PERTURBATION_PERTURBEDclass-attributeinstance-attribute

Which types of Instances we are evaluating, to be populated during metric evaluation. PERTURBATION_PERTURBED (default) means we are evaluating on perturbed instances, PERTURBATION_ORIGINAL means we are evaluating the unperturbed version of instances where this perturbation applies, and, PERTURBATION_WORST means the the minimum metric between the two.

### seed: Optional[int] = Noneclass-attributeinstance-attribute

Seed added to instance_id when generating perturbation

## Requestdataclass

A`Request` specifies how to query a language model (given a prompt, complete it). It is the unified representation for communicating with various APIs (e.g., GPT-3, Jurassic).

### model_deployment: str = ''class-attributeinstance-attribute

Which model deployment to query -> Determines the Client. Refers to a deployment in the model deployment registry.

### model: str = ''class-attributeinstance-attribute

Which model to use -> Determines the Engine. Refers to a model metadata in the model registry.

### embedding: bool = Falseclass-attributeinstance-attribute

Whether to query embedding instead of text response

### prompt: str = ''class-attributeinstance-attribute

What prompt do condition the language model on

### temperature: float = 1.0class-attributeinstance-attribute

Temperature parameter that governs diversity

### num_completions: int = 1class-attributeinstance-attribute

Generate this many completions (by sampling from the model)

### top_k_per_token: int = 1class-attributeinstance-attribute

Take this many highest probability candidates per token in the completion

### max_tokens: int = 100class-attributeinstance-attribute

Maximum number of tokens to generate (per completion)

### stop_sequences: List[str] = field(default_factory=list)class-attributeinstance-attribute

Stop generating once we hit one of these strings.

### echo_prompt: bool = Falseclass-attributeinstance-attribute

Should`prompt` be included as a prefix of each completion? (e.g., for evaluating perplexity of the prompt)

### top_p: float = 1class-attributeinstance-attribute

Same from tokens that occupy this probability mass (nucleus sampling)

### presence_penalty: float = 0class-attributeinstance-attribute

Penalize repetition (OpenAI & Writer only)

### frequency_penalty: float = 0class-attributeinstance-attribute

Penalize repetition (OpenAI & Writer only)

### random: Optional[str] = Noneclass-attributeinstance-attribute

Used to control randomness. Expect different responses for the same request but with different values for`random`.

### messages: Optional[List[Dict[str, str]]] = Noneclass-attributeinstance-attribute

Used for chat models. (OpenAI only for now). if messages is specified for a chat model, the prompt is ignored. Otherwise, the client should convert the prompt into a message.

### multimodal_prompt: Optional[MultimediaObject] = Noneclass-attributeinstance-attribute

Multimodal prompt with media objects interleaved (e.g., text, video, image, text, ...)

### image_generation_parameters: Optional[ImageGenerationParameters] = Noneclass-attributeinstance-attribute

Parameters for image generation.

### response_format: Optional[ResponseFormat] = Noneclass-attributeinstance-attribute

EXPERIMENTAL: Response format. Currently only supported by OpenAI and Together.

### model_host: strproperty

Returns the model host (referring to the deployment). Not to be confused with the model creator organization (referring to the model).

'openai/davinci' => 'openai'

'together/bloom' => 'together'

### model_engine: strproperty

Returns the model engine (referring to the model). This is often the same as self.model_deploymentl.split("/")[1], but not always. For example, one model could be served on several servers (each with a different model_deployment) In that case we would have for example: 'aws/bloom-1', 'aws/bloom-2', 'aws/bloom-3' => 'bloom' This is why we need to keep track of the model engine with the model metadata. Example: 'openai/davinci' => 'davinci'

### validate()

## RequestResultdataclass

What comes back due to a`Request`.

### success: boolinstance-attribute

Whether the request was successful

### embedding: List[float]instance-attribute

Fixed dimensional embedding corresponding to the entire prompt

### completions: List[GeneratedOutput]instance-attribute

List of completion

### cached: boolinstance-attribute

Whether the request was actually cached

### request_time: Optional[float] = Noneclass-attributeinstance-attribute

How long the request took in seconds

### request_datetime: Optional[int] = Noneclass-attributeinstance-attribute

When was the request sent? We keep track of when the request was made because the underlying model or inference procedure backing the API might change over time. The integer represents the current time in seconds since the Epoch (January 1, 1970).

### error: Optional[str] = Noneclass-attributeinstance-attribute

If`success` is false, what was the error?

### error_flags: Optional[ErrorFlags] = Noneclass-attributeinstance-attribute

Describes how to treat errors in the request.

### batch_size: Optional[int] = Noneclass-attributeinstance-attribute

Batch size (`TogetherClient` only)

### batch_request_time: Optional[float] = Noneclass-attributeinstance-attribute

How long it took to process the batch? (`TogetherClient` only)

### render_lines() -> List[str]

## PerInstanceStatsdataclass

Captures a unit of evaluation.

### instance_id: strinstance-attribute

### perturbation: Optional[PerturbationDescription]instance-attribute

### train_trial_index: intinstance-attribute

Which replication

### stats: List[Stat]instance-attribute

Statistics computed from the predicted output

## Statdataclass

A mutable class that allows us to aggregate values and report mean/stddev.

### name: MetricNameinstance-attribute

### count: int = 0class-attributeinstance-attribute

### sum: float = 0class-attributeinstance-attribute

### sum_squared: float = 0class-attributeinstance-attribute

### min: Optional[float] = Noneclass-attributeinstance-attribute

### max: Optional[float] = Noneclass-attributeinstance-attribute

### mean: Optional[float] = Noneclass-attributeinstance-attribute

### variance: Optional[float] = Noneclass-attributeinstance-attribute

This is the population variance, not the sample variance.

See [https://towardsdatascience.com/variance-sample-vs-population-3ddbd29e498a](https://towardsdatascience.com/variance-sample-vs-population-3ddbd29e498a) for details.

### stddev: Optional[float] = Noneclass-attributeinstance-attribute

This is the population standard deviation, not the sample standard deviation.

See [https://towardsdatascience.com/variance-sample-vs-population-3ddbd29e498a](https://towardsdatascience.com/variance-sample-vs-population-3ddbd29e498a) for details.

### add(x) -> Stat

### merge(other: Stat) -> Stat

### __repr__()

### bare_str() -> str

### take_mean()

Return a version of the stat that only has the mean.
