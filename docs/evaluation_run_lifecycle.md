# Evaluation Run Lifecycle

Each invocation of `helm-run` will run a number of **evaluation runs**. Each evaluation run uses a single scenario and a single model, and is performed independently form other evaluation runs. Evaluation runs are usually executed serially / one at a time by the default runner, though some alternate runners (e.g. `SlurmRunner`) may execute evaluation runs in parallel.

## Evaluation steps

An evaluation run has the following steps:

1. Get in-context learning and evaluation instances from scenario. Each instance has an input (e.g. question) and a set of reference outputs (e.g. multiple choice options).
2. (Advanced) Run _data augmenters / perturbations_ on the base instances to generate perturbed instances.
3. Perform _adaptation_ to transform the in-context learning instances and evaluation instances into model inference requests, which contain prompts and other request parameters such as request temperature and stop sequences.
4. Send the requests to the models and receives the request responses.
5. Compute the per-instance stats and aggregate them to per-run stats.

The following code and data objects are responsible involved in an evaluation run:

1. A `Scenario` provides the in context learning and evaluation `Instance`s.
2. A `DataAugmenter` takes in base `Instance` and generates perturbed `Instance`.
3. A `Adapter` transforms in-context learning instances and evaluation instances into model inference `Request`s.
4. A `Client` sends the `Requests` to the models and receives `RequestResponse`s.
5. `Metrics`s take in `RequestState`s (which each contain a `Instance`, `Request`,`RequestResponse`, and additional instance context) and compute aggregated adn per-instanace `Stat`s.

## Specifications

Each evaluation run is _fully specified_ by a **run specification** (`RunSpec`), which contains a specification for each of the above code objects (_except_ `ClientSpec`, which is a special case):

1. A `ScenarioSpec` specifies a `Scenario` instances.
2. A `DataAugmenterSpec` specifies a `DataAugmenter` instance.
3. An `AdapterSpec` specifies an `Adapter` instance.
4. `MetricSpec`s specifies `Metric` instances.

Note: The `RunSpec` does not contain a `ClientSpec` specifies the `Client` instance. Instead, the `RunSpec` specifies the name of the model deployment inside `AdapterSpec`. During the evaluation run, the model deployment name is used to retreive the `ClientSpec` from built-in or user-provided model deployment configurations, which is then used to construct the `Client`. This late binding allows the HELM user to perform user-specific configuration of clients, such as changing the type or location of the model inference platform for the model.

## Serializability

The objects above can be grouped into three categories:

1. Specifications (`RunSpec`, `ScenarioSpec`, `DataAugmenterSpec`, `AdapterSpec`, `ClientSpec`, and `MetricsSpec`) are serializable. They may be written to evaluation run output files, to provide a record of how the evaluation run was configured and how to reproduce it.
2. Code objects (`Scenario`, `DataAugmenter`, `Adapter`, `Client`, `Metric`) are _not_ serializable. These contain program logic used for by the evlauation run. Users can implement custom subclasses of these objects if needed.
3. Data objects (`Instance`, `Request`, `Response`, `Stat`) are serializable. These are typcically produced as outputs of code objects and written to the evaluation run output files.

## Run spec functions

When a user runs `helm-run`, the evaluation runner will perform a number of evaluation runs, each specified by a `RunSpec`. However, the user typically does not provide the `RunSpec`s directly. Instead, the `RunSpec`s are produced by **run spec functions**. The user instead passes one or more **run entries** to `helm-run`, which are short strings (e.g. `mmlu:subject=anatomy,model=openai/gpt2`) that specify how to invoke the run spec functions to get the actual `RunSpec`s.

The run entry format is explained further on its own documentation.
