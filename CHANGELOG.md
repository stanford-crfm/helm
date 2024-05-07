# Changelog

## [Upcoming]

## [v0.5.1] - 2024-05-06

### Scenarios

- Updated VLM scenarios for VHELM (#1592)
- Added `trust_remote_code=True` for `math` and `legalbench` scenarios (#2597)

### Models

- Added Google Gemini 1.5 Pro as a VLM (#2607)
- Add Mixtral 8x22b Instruct, Llama 3 Chat, and Yi Chat (#2610)
- Updated VLM models for VHELM (#1592)
- Improved handling of Gemini content blocking (#2603)
- Added default instructions prefix for multiple-choice joint adaptation for Claude 3
- Renamed some model names for consistency (#2596)
- Added Snowflake Arctic Instruct (#2599, #2591)

### Frontend

- Added global landing page (#2593)

### Evaluation Results

- [VHELM v1.0.0](https://crfm.stanford.edu/helm/mmlu/v1.0.0/)
    - Initial release with Gemini 1.5 Pro, Claude 3 Sonnet, Claude 3 Opus, Gemini 1.0 Pro Vision, IDEFICS 2 and GPT-4V

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @farzaank
- @neelguha
- @percyliang
- @teetone
- @yifanmai

## [v0.5.0] - 2024-04-23

### Breaking changes

- The `--run-specs` flag was renamed to `--run-entries` (#2404)
- The `run_specs*.conf` files were renamed to `run_entries*.conf` (#2430)
- The `model_metadata` field was removed from `schema*.yaml` files (#2195)
- The `helm.proxy.clients` package was moved to `helm.clients` (#2413)
- The `helm.proxy.tokenizers` package was moved to `helm.tokenizers` (#2403)
- The frontend only supports JSON output produced by `helm-summarize` at version 0.3.0 or newer (#2455)
- The `Sequence` class was renamed to `GeneratedOutput` (#2551)
- The `black` linter was upgraded from 22.10.0 to 24.3.0, which produces different output  - run `pip install --upgrade black==24.3.0` to upgrade this dependency (#2545)
- The `anthropic` dependency was upgraded from `anthropic~=0.2.5` to `anthropic~=0.17` - run `pip install --upgrade anthropic~=0.17` to upgrade this dependency (#2432)
- The `openai` dependency was upgraded from `openai~=0.27.8` to `openai~=1.0`- run `pip install --upgrade openai~=1.0` to upgrade this dependency (#2384)
    - The SQLite cache is not compatible across this dependency upgrade - if you encounter an `ModuleNotFoundError: No module named 'openai.openai_object'` error after upgrading `openai`, you will have to delete your old OpenAI SQLite cache (e.g. by running `rm prod_env/cache/openai.sqlite`)

### Scenarios

- Added DecodingTrust (#1827)
- Added Hateful Memes (#1992)
- Added MMMU (#2259)
- Added Image2Structure (#2267, #2472)
- Added MMU (#2259)
- Added LMEntry (#1694)
- Added Unicorn vision-language scenario (#2456)
- Added Bingo vision-language scenario (#2456)
- Added MultipanelVQA (#2517)
- Added POPE (#2517)
- Added MuliMedQA (#2524)
- Added ThaiExam (#2534)
- Added Seed-Bench and MME (#2559)
- Added Mementos vision-language scenario (#2555)
- Added Unitxt integration (#2442, #2553)

### Models

- Added OpenAI gpt-3.5-turbo-1106, gpt-3.5-turbo-0125, gpt-4-vision-preview, gpt-4-0125-preview, and gpt-3.5-turbo-instruct (#2189, #2295, #2376, #2400)
- Added Google Gemini 1.0, Gemini 1.5, and Gemini Vision (#2186, #2189, #2561)
- Improved handling of content blocking in the Vertex AI client (#2546, #2313)
- Added Claude 3 (#2432, #2440, #2536)
- Added Mistral Small, Medium and Large (#2307, #2333, #2399)
- Added Mixtral 8x7b Instruct and 8x22B (#2416, #2562)
- Added Luminous Multimodal (#2189)
- Added Llava and BakLava (#2234)
- Added Phi-2 (#2338)
- Added Qwen1.5 (#2338, #2369)
- Added Qwen VL and VL Chat (#2428)
- Added Amazon Titan (#2165)
- Added Google Gemma (#2397)
- Added OpenFlamingo (#2237)
- Removed logprobs from models hosted on Together (#2325)
- Added support for vLLM (#2402)
- Added DeepSeek LLM 67B Chat (#2563)
- Added Llama 3 (#2579)
- Added DBRX Instruct (#2585)

### Framework

- Added support for text-to-image models (#1939)
- Refactored of `Metric` class structure (#2170, #2171, #2218)
- Fixed bug in computing general metrics (#2172)
- Added a `--disable-cache` flag to disable caching in `helm-run` (#2143)
- Added a `--schema-path` flag to support user-provided `schema.yaml` files in `helm-summarize` (#2520)

### Frontend

- Switched to the new React frontend for local development by default (#2251)
- Added support for displaying images (#2371)
- Made various improvements to project and version dropdown menus (#2272, #2401, #2458)
- Made row and column headers sticky in leaderboard tables (#2273, #2275)

### Evaluation Results

- [Lite v1.1.0](https://crfm.stanford.edu/helm/lite/v1.1.0/)
    - Added results for Phi-2 and Mistral Medium
- [Lite v1.2.0](https://crfm.stanford.edu/helm/lite/v1.2.0/)
    - Added results for Llama 3, Mixtral 8x22B, OLMo, Qwen1.5, and Gemma
- [HEIM v1.1.0](https://crfm.stanford.edu/helm/heim/v1.1.0/)
    - Added results for Adobe GigaGAN and DeepFloyd IF
- [Instruct v1.0.0](https://crfm.stanford.edu/helm/instruct/v1.0.0/)
    - Initial release with results for OpenAI GPT-4, OpenAI GPT-3.5 Turbo, Anthropic Claude v1.3, Cohere Command beta
- [MMLU v1.0.0](https://crfm.stanford.edu/helm/mmlu/v1.0.0/)
    - Initial release with 22 models
- [MMLU v1.1.0](https://crfm.stanford.edu/helm/mmlu/v1.1.0/)
    - Added results for Llama 3, Mixtral 8x22B, OLMo, and Qwen1.5 (32B)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @acphile
- @akashc1
- @AlphaPav
- @andyzorigin
- @boxin-wbx
- @brianwgoldman
- @chenweixin107
- @danielz02
- @elronbandel
- @farzaank
- @garyxcj
- @ImKeTT
- @JosselinSomervilleRoberts
- @kangmintong
- @michiyasunaga
- @mmonfort
- @mtake
- @percyliang
- @polaris-73
- @pongib
- @ritik99
- @ruixin31
- @sbdzdz
- @shenmishajing
- @teetone
- @tybrs
- @YianZhang
- @yifanmai
- @yoavkatz

## [v0.4.0] - 2023-12-20

### Models

- Added Google PaLM 2 (#2087, #2111, #2139)
- Added Anthropic Claude 2.1 and Claude Instant 1.2 (#2095, #2123)
- Added Writer Palmyra-X v2 and v3 (#2104)
- Added OpenAI GPT-4 Turbo preview (#2092)
- Added 01.AI Yi (#2009)
- Added Mistral AI Mixtral-8x7B (#2130)
- Fixed race condition with "Already borrowed" error for  Hugging Face tokenizers (#2088, #2091, #2116)
- Support configuration precision and quantization in HuggingFaceClient (#1912)
- Support LanguageModelingAdapter for HuggingFaceClient (#1964)

### Scenarios

- Added VizWiz Scenario (#1983)
- Added LegalBench scenario (#2129)
- Refactored CommonSenseScenario into HellaSwagScenario, OpenBookQA, SiqaScenario, and PiqaScenario (#2117, #2118, #2119)
- Added run specs configuration for HELM Lite (#2009)
- Changed the default metric in GSM8K to check exact match of the final number in the response (#2130)

### Framework

- Added tutorial for computing the leaderboard rank of a model using the method from "Efficient Benchmarking (of Language Models)" (#1968, #1986, #1985)
- Refactored ModelMetadata, ModelDeployment and Tokenizer, and moved configuration to YAML files (#1903, #1994)
- Fixed a bug regarding writing `runs_to_run_suites.json` when using `helm-release` with `--release` (#2012)
- Made pymongo an optional dependency (#1882)
- Made SlurmRunner retry some failed Slurm requests (#2077)
- Shortened cache retry time (#2081)
- Added retrying to AutoTokenizer (#2090)
- Added support for user configuration of model deployments and tokenizer configurations (#1996, #2142)
- Added support for passing in an arbitrary schema file to `helm-rummarize` (#2075)
- Changed the prompt format for some instruction following models (#2130)
- Added py.typed to package type information (#2169)

### Frontend

- Made visual improvements and bugfixes for the new React frontend (#1947, #2000, #2005, #2018)
- Changed front page on Raect frontend to display a mini leaderboard (#2113, #2128)
- Added a dropdown menu for switching between different HELM results websites (#1947)
- Added a dropdown menu for switching between different versions (#2135)

### Evaluation Results

- Launched new React frontend
- [HELM Classic v0.4.0](https://crfm.stanford.edu/helm/classic/v0.4.0/)
    - Added evaluation results for Mistral
- [HELM Lite v1.0.0](https://crfm.stanford.edu/helm/lite/v1.0.0/)
    - Launched new [HELM Lite leaderboard](https://crfm.stanford.edu/2023/12/19/helm-lite.html) with 30 models and 10 scenarios

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @brianwgoldman
- @dlwh
- @farzaank
- @JosselinSomervilleRoberts
- @krh26
- @neelguha
- @percyliang
- @perlitz
- @pettter
- @ruixin31
- @teetone
- @yifanmai
- @yotamp

## [v0.3.0] - 2023-11-01

### Models

- Added support for Lit-GPT (#1792)
- Added support for stop sequences in HuggingFaceClient (#1892, #1909)
- Added Mistral 7B model (#1906)
- Added IDEFICS model (#1871)
- Added Anthropic Claude 2 (#1900)

### Scenarios

- Added 31 scenarios from [CLEVA](https://arxiv.org/abs/2308.04813) for evaluation of Chinese language models (#1824, #1864)
- Added VQA scenario model (#1871)
- Adddd support for running MCQA scenarios from users' JSONL files (#1889)

### Metrics

- Fixed a bug that prevented using Anthropic Claude for model critique (#1862)

### Frontend

- Added a React frontend (#1819, #1893)

### Framework

- Added support for multi-modal scenarios and Vision Language Model (VLM) evaluation (#1871)
- Added support for Python 3.9 and 3.10 (#1897)
- Added a new `Tokenizer` class in preparation for removing `tokenize()` and `decode()` from `Client` in a future release (#1874)
- Made more dependencies optional instead of required, and added install command suggestions (#1834, #1961)
- Added support for configuring users' model deployments through YAML configuration files (#1861)

### Evaluation Results

- Added evaluation results for Stanford Alpaca, MosaicML MPT, TII UAE Falcon, LMSYS Vicuna

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @aniketmaurya
- @Anindyadeep
- @brianwgoldman
- @drisspg
- @farzaank
- @fzyxh
- @HenryHZY
- @Jianqiao-Zhao
- @JosselinSomervilleRoberts
- @LoryPack
- @lyy1994
- @mkly
- @msaroufim
- @percyliang
- @RossBencina
- @teetone
- @yifanmai
- @zd11024

## [v0.2.4] - 2023-09-20

### Models

- Added Meta LLaMA, Meta Llama 2, EleutherAI Pythia, Together RedPajama on Together (#1821)
- Removed the unofficial chat-gpt client in favor of the official API (#1809)
- Added support for models for the NeurIPS Efficiency Challenge (#1693)

### Frontend

- Added support for rendering train-test overlap stats in the frontend (#1747)
- Fixed a bug where stats with NaN values would cause the frontend to fail to render tables (#1784)

### Framework

- Moved many dependencies, especially those only used by a single model provider or a small number of runs, to optional extra dependencies (#1798, #1844)
- Widened some dependencies (e.g. PyTorch) to reduce dependency conflicts with other packages  (#1759)
- Added `MaxEvalInstancesRunExpander` to allow overriding the number of eval instances at the run level (#1837)
- Updated human critique evaluation on Amazon Mechanical Turk to support emoji and other special characters (#1773)
- Fixed a bug where in-context learning examples with multiple correct references were adapted to prompts where all the correct references are concatenated together as the output, which was not intended for some scenarios (e.g. narrative_qa, natural_qa, quac and wikifact) (#1785)
- Fixed a bug where ObjectSpec is not hashable if any arg is a list (#1771)

### Evaluations

- Added evaluation results for Meta LLaMA, Meta Llama 2, EleutherAI Pythia, Together RedPajama on Together
- Corrected evaluation results for AI21 Jurassic-2 and Writer Palmyra for the scenarios narrative_qa, natural_qa, quac and wikifact, as they were affected by the bug fixed by #1785

### Contributors

Thank you to the following contributors for your contributions to this HELM release!

- @AndrewJGaut
- @andyzorigin
- @bidyapati-p
- @drisspg
- @mkly
- @msaroufim
- @percyliang
- @teetone
- @timothylimyl
- @unnawut
- @yifanmai

## [v0.2.3] - 2023-07-25

### Models

- Added BigCode StarCoder (#1506) 
- Added OPT 1.3B and 6.7B (#1468)
- Added OpenAI gpt-3.5-turbo-0613 (#1667), gpt-3.5-turbo-16k-0613, gpt-4-0613, gpt-4-32k-0613 (#1468), gpt-4-32k-0314, gpt-4-32k-0314 (#1457)
- Added OpenAI text-embedding-ada-002 (#1711)
- Added Writer Palmyra (#1669,  #1491)
- Added Anthropic Claude (#1484)
- Added Databricks Koala on Together (#1701)
- Added Stability AI StableLM and Together RedPajama on Together

### Scenarios

- Added legal summarization scenarios (#1454)
- Fixed corner cases in window service truncation (#1449)
- Pinned file order for ICE, APPS (code) and ICE scenarios (#1352)
- Fixed random seed for entity matching scenario (#1475)
- Added Spider text-to-SQL (#1385)
- Added Vicuna scenario (#1641), Koala scenario (#1642), open_assistant scenario (#1622), and Anthropic-HH-RLHF scenario (#1643) for instruction-following
- Added verifiability judgement scenario (#1518)

### Metrics

- Fixed bug in multi-choice exact match calculation when scores are tied (#1494)

### Framework

- Added script for estimating the cost of a run suite (#1480)
- Added support for human critique evaluation using Surge AI (#1330), Scale AI (#1609), and Amazon Mechanical Turk (#1539)
- Added support for LLM critique evaluation (#1627)
- Decreased running time of helm-summarize (#1716)
- Added `SlurmRunner` for distributing `helm-run` jobs over Slurm (#1550)
- Migrated to the `setuptools.build_meta` backend (#1535)
- Stopped non-retriable errors (e.g. content filter errors) from being retried (#1533)
- Added logging for stack trace and exception message when retries occur (#1555)
- Added file locking for `ensure_file_downloaded()` (#1692)

## Evaluations

- Added evaluation results for AI21 Jurassic-2 and Writer Palmyra

## [v0.2.2] - 2023-03-30

### Models

- Added Cohere Command (#1321)
- Added Flan-T5 (#1398)
- Added H3 (#1398)
- Added GPT-NeoXT-Chat-Base-20B (#1407)
- Added OpenAI gpt-3.5-turbo-0301 (#1401)
- Added AI21 Jurassic-2 models (#1409)

### Scenarios

- Some improvements to LEXTREME and LexGLUE legal scenarios (#1429)
- Added OpinionsQA scenario (#1424)

### Metrics

- Added multilabel classification metrics (#1408)

### Framework

- Fixed `--exit-on-error` not working and added `--skip-completed-runs` (#1400)
- Disabled tqdm in non-interactive mode (#1351)
- Added plotting (#1403, #1411)
- Added Hugging Face Model Hub integration (#1103)

### Evaluations

- Added evaluation results for Cohere Command and Aleph Alpha Luminous

## [v0.2.1] - 2022-02-24

### Models

- Added BigCode SantaCoder (#1312)

### Scenarios

- Added LEXTREME and LexGLUE legal scenarios (#1216)
- Added WMT14 machine translation scenario (#1329)
- Added biomedical scenarios: COVID Dialogue, MeQSum, MedDialog, MedMCQA, MedParagraphSimplification, MedQA, PubMedQA (#1332)

### Framework

- Added `--run-specs` flag to `helm-run` (#1302)
- Reduced running time of `helm-summarize` (#1269)
- Added classification metrics (#1368)
- Updated released JSON assets to conform to current JSON schema

## [v0.2.0] 2022-01-11

### Models

- Added Aeph Alpha's Luminous models (#1215)
- Added AI21's J1-Grande v2 beta model (#1177)
- Added OpenAI's ChatGPT model (#1231)
- Added OpenAI's text-davinci-003 model (#1200)

### Scenarios

- Added filtering by subject and level for MATHScenario (#1137)

### Frontend

- Reduced frontend JSON file sizes (#1185)
- Added table sorting in frontend (#832)
- Fixed frontend bugs for certain adapter methods (#1236, #1237)
- Fixed frontend bugs for runs with multiple trials (#1211)

### Adaptation

- Improved sampling of in-context examples (#1172)
- Internal refactor (#1280)

## Result summarization

- Added average win-rate computation for model-v-scenario tables (#1240)
- Added additional calibration metrics as a "Targeted evaluation" (#1247)

### Misc

- Added documentation to Read the Docs (#1159, #1164)
- Breaking schema change: `input` of `Instance` and `output` of `Reference` are now objects (#1280)

## [v0.1.0] - 2022-11-17

- Initial release

[upcoming]: https://github.com/stanford-crfm/helm/compare/v0.5.1...HEAD
[v0.5.1]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.1
[v0.5.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.0
[v0.4.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.4.0
[v0.3.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.3.0
[v0.2.4]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.4
[v0.2.3]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.3
[v0.2.2]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.2
[v0.2.1]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.1
[v0.2.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.0
[v0.1.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.1.0
