# Changelog

## [Upcoming]

## [v0.5.11] - 2025-12-04

### Breaking Changes

- Remove deprecated Claude 3.5 model from LLM juries (#3926)
- Switch Gemini 2.5 to use `GoogleGenAIClient` (#3944)

### Models

- Add Claude Haiku 4.5 (#3922)
- Add GPT-5.1 model (#3927, 3937)
- Add `GoogleGenAIClient` (#3925)
- Add Gemini 3 Pro (#3936, 3937)
- Add Gemini Robotics-ER 1.5 (#3939)
- Add Kimi K2 model on Together (#3943)
- Switch Gemini 2.5 to use `GoogleGenAIClient` (#3944)
- Add more Qwen3 and Qwen3-Next models on Together AI (#3947)

### Scenarios

- Convert AraTrust to use multiple_choice_joint adapter (#3920)
- Remove deprecated Claude 3.5 model from LLM juries (#3926)
- Remove stop sequences in ALRAGE scenario (#3948)

### Framework

- Add support for uv (#3929, #3932)
- Support schema files in JSON format in helm-summarize (#3924)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @beyandbey
- @chaudhariatul
- @kiro-agent
- @yifanmai

## [v0.5.10] - 2025-11-10

### Breaking Changes

- Python 3.9 is no longer supported as it reached end-of-life in October 2025. (#3903)

### Models

- Add Palmyra X5 on Bedrock (#3905)
- Add IBM Granite 4 models on Hugging Face (#3906)
- Fix incorrect Arabic model configs (#3907)

### Scenarios

- Fix `shc_privacy_med` scenario in MedHELM (#3911)

### Framework

- Remove Python 3.9 support (#3903)
- Separate the definition of the ArgumentParser from its usage (#3908)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @asad-aali
- @aunell
- @Erotemic
- @MiguelAFH
- @patelfagun1998
- @yifanmai

## [v0.5.9] - 2025-10-01

### Models

- Add Arabic language models
    - AceGPT-v2 (#3867)
    - ALLaM (#3867)
    - SILMA (#3867)
    - Jais Family (#3878)
    - Jais Adapted (#3881)
- Add Qwen3-Next 80B A3B Thinking model (#3875)
- Add DeepSeek-R1-Distill-Llama-70B and DeepSeek-R1-Distill-Qwen-14B (#3873)
- Add Qwen3-Next 80B A3B Thinking model (#3891)
- Add Falcon3 models (#3894)
- Add Claude 4.5 Sonnet (#3897)

### Scenarios

- Fix breakages in `shc_privacy_med` and `shc_proxy_med` (#3876)
- Allow applying regular expression pattern before mapping output (#3882)
- Add output mapping patterns for Arabic MCQA scenarios (#3885)
- Update speech language pathology scenarios to use Hugging Face Datasets (#3835)

### Framework

- Make priority optional for run entries (#3865)
- Add `herror()` and `hexception()` to logger (#3789)
- Log error stack traces from within various clients (#3880)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @chakravarthik27
- @MiguelAFH
- @patelfagun1998
- @yifanmai

## [v0.5.8] - 2025-08-29

### Models

- Add GLM-4.5-AIR-FP8 model (#3785)
- Add Qwen3 235B A22B Instruct 2507 FP8 (#3788)
- Add Gemini 2.5 Flash-Lite GA (#3776)
- Add gpt-oss (#3789, #3794)
- Add GPT-5 (#3793, #3797)
- Handle safety and usage guidelines errors from Grok API (#3770)
- Handle Gemini responses with max tokens reached during thinking (#3804)
- Add OpenRouterClient (#3811)

### Scenarios

- Fix instructions and prompt formatting for InfiniteBench En.MC (#3790)
- Add MedQA and MedMCQA to MedHELM (#3781)
- Add or modify Arabic language scenarios:
    - ALRAGE (#3806, #3721, #3831, #3831)
    - MadinahQA (#3806, #3800, #3817)
    - ArabicMMLU (#3806, #3817, #3838)
    - AraTrust (#3806, #3819)
    - Arabic EXAMS (#3806, #3818, #3825)
    - AlGhafa (#3806, #3821)
    - MBZUAI Human-Translated Arabic MMLU (#3822)
- Add run expander for Arabic language instructions for Arabic MCQA scenarios (#3833)
- Allow configuration of LLM-as-a-judge models in MedHELM scenarios (#3812)
- Add user-configurable MedHELM scenario (#3844)

### Frontend

- Display Arabic text in RTL direction in frontend (#3807)
- Fix regular expression query handling in run predictions (#3826)
- Fix invalid sort column index error in leaderboard (#3845)

### Framework

- Migrate to pyproject.toml (#3767)
- Various fixes for proxy server (#3801, #3802, #3803)
- Raise error if helm-summarize is given a non-existent suite (#3805)
- Allow setting reference prefix characters (#3809)
- Auto-generate schema in helm-summarize if `--auto-generate-schema` is specified (#3813, #3814, #3828, #3839, #3842, #3848, #3850)
- Omit empty tables for metric groups in helm-summarize (#3851)
- Add `get_metadata()` method for many scenarios and metrics (#3815, #3829, #3832, #3834, #3841, #3843, #3849, #3840, #3830)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @chakravarthik27
- @Erotemic
- @MiguelAFH
- @patelfagun1998
- @yifanmai

## [v0.5.7] - 2025-07-31

### Breaking Changes

- In v0.5.6, `HuggingFaceClient` would call `apply_chat_template` with `add_generation_prompt()` unspecified, which caused it to default to `False`. In v0.5.7, `HuggingFaceClient` now calls `add_generation_prompt()` with `apply_chat_template=True`. (#3703)
- Previously, in the `model_deployments.yaml` configuration files, when determining the default model deployment for a model, the first model deployment in the file would be preferred, and model deployments in the package would be preferred over those in `prod_env/`. Now, the last model deployment in the file will be preferred, and model deployments in `prod_env` will be preferred over those in the package. (#3694)

### Models

- Add o3-pro (#3671)
- Allow passing initialization kwargs to VLLMClient and OpenAIClient (#3690)
- Add support for Granite thinking on vLLM (#3692)
- Add some Brazilian models (#3686)
- Always set `add_generation_prompt` to `True` in `apply_chat_template` in `HuggingFaceClient` (#3703)
- Add support for chat models on vLLM (#3729)
- Add Kimi K2 (#3758)
- Add Grok 4 (#3759)

### Scenarios

- Add LMKT (Language model cultural alignment transfer) (#3682)
- Modify SLPHelm (#3679)
- Add InfiniteBench En.MC (#3687)
- Add MMMLU (#3699)
- Add Arabic MMLU, AlGhafa (#3700)
- Add AlGhafa (#3706)
- Add EXAMS (#3714)
- Add AraTrust (#3715)
- Add HEALTHQA-BR (#3711)
- Add BLEUX (#3711)
- Remove numeracy scenario (#3733)
- Fixes to HEIM (#3765)

### Framework

- Improvements to logging (#3683, #3696, #3744)
- Change logic for default model deployments (#3694)
- Loosen various dependency version specifiers (#3725, #3726)
- Add short_description field to RunGroup in schema (#3755)
- Add support for configuring the number of retries using environment variables (#3766)

### Frontend

- Show run group name and description from schema on run page (#3753)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @Erotemic
- @Gum-Joe
- @IriedsonSouto
- @Kazf28
- @ledong0110
- @lucas-s-p
- @martinakaduc
- @MiguelAFH
- @RonalddMatias
- @samirunni
- @sangttruong
- @sashimono-san
- @yifanmai

## [v0.5.6] - 2025-06-20

### Breaking Changes

- Previously, HuggingFaceClient did not apply the chat template for chat models. `HuggingFaceClient` will now automatically apply the chat template if the tokenizer contains one. This may result in incorrect behavior for some non-chat models that have tokenizers with chat templates e.g. Qwen2, Qwen 2.5. For these models, the default behavior can be overridden by explicitly setting `apply_chat_template` in `args`. (#3678)

### Models

- Fixed bug regarding `reasoning_effort` not being set correct for OpenAI models (#3501)
- Add Llama 4 Scout and Maverick on Together AI (#3504)
- Add Qwen2.5 Omni 7B (#3505, #3530)
- Add Grok 3 models (#3518, #3529, #3534)
- Add GPT-4.1 models (#3527, #3528)
- Add Gemini 2.5 models (#3538, #3614)
- Add OpenAI o3 and o4 series models (#3539, #3617)
- Adjust max_tokens restriction for Anthropic models (#3555)
- Add Vietnamese LLMs (#3563)
- Add Qwen3 235B on Together (#3579)
- Add Palmyra X5 (#3580)
- Add COREBench v0 audio scenario (#3567, #3610)
- Add Anthropic Claude 4 models (#3607)
- Add extended thinking to Claude 3.7 and Claude 4 (#3622)
- Support streaming in Anthropic client (#3624)
- Added support for reasoning content (#3632, #3650)
- Add DeepSeek-R1-0528 on Together AI (#3636)
- Add Amazon Nova Premier model (#3656)
- Add Marin 8B Instruct (#3658, #3666)
- Add HuggingFacePipelineClient (#3665)
- Add SmolLM2 (#3665, #3668)
- Add OLMo 2 and OLMoE models (#3667, #3676)
- Add Granite 3.3 8B Instruct (#3675)
- Add Mistral Medium 3 (#3674)
- Add Gemini 2.5 Flash-Lite (#3677)
- Apply chat template in HuggingFaceClient and make chat template usage configurable (#3678)
- Add support for OpenAI Responses API (#3652)

### Scenarios

- Improvements to VHELM (#3500)
- Improvements to audio scenarios (#3500, #3523,)
- Improvements to MedHELM scnearios (#3507, #3499, #3506, #3496, #3509, #3503, #3522, #3508, #3540, #3535, #3548, #3545, #3560, #3565, #3557, #3566, #3577, #3625, #3631)
- Add Vietnamese language scenarios (#3464, #3511, #3594, #3536, #3556, #3551, #3611, #3634)
- Add KPI Edgar scneario (#3495)
- Add run entries with additional instructions for MMLU-Winogrande-Afr (#3497)
- Add run entries for reasoning models (#3541, #3544)
- Fix bug with AIR-Bench handling of empty responses (#3543)
- Fix bug regarding the MMLU-Pro computer science subject (#3547)
- Add InfiniteBench En.QA (#3588)
- Add speech pathology scenarios (#3553, #3558)
- Bugfix for Image2Struct: EMS metric fails when the inputs are identical homogenous images files (#3576)
- Fix download URL for TruthfulQA CSV file (#3601)
- Use Python logging library in hlog (#3598, #3606)
- Add OpenAI-MRCR scenario (#3653)
- Rename InfiniteBench Sum to InfiniteBench En.Sum (#3657)
- Change metrics for long context scenarios (#3659)

### Framework

- Avoid accessing SQLite accounts file in helm-run (#3573)
- Support video understanding (#3578)
- Use Python logging library (#3598)

### Frontend

- Support displaying nested objects in annotations (#3513)
- Add displaying nested objects in annotations (#3512)
- Fix mobile menu z-index (#3514)
- Change favicon URL (#3597)
- Set MIME type for config.js in server (#3605)
- Fixes pagination of metrics displayed on the front-end (#3646)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @asillycat
- @aunell
- @chiheem
- @Erotemic
- @Gum-Joe
- @haoqiu1
- @HennyJie
- @ImKeTT
- @jmbanda
- @Joey-Ji
- @lucas-s-p
- @martinakaduc
- @MiguelAFH
- @mtake
- @patelfagun1998
- @raulista1997
- @ryokawajp
- @sangttruong
- @suhana13
- @tcz
- @teetone
- @yifanmai

## [v0.5.5] - 2025-04-04

### Breaking Changes

- Optimum Intel OpenVINO is no longer supported (#3153)

### Scenarios

- HELM Capabilities scenarios
    - Add GPQA scenario (#3068, #3078, #3079, #3100, #3096, #3420, #3445)
    - Add MMLU-Pro scenario (#3108, #3125, #3200, #3272, #3458)
    - Add IFEval scenario (#3122, #3275)
    - Add WildBench (#3150, #3283, #3299, #3339, #3318, #3360)#3476
    - Add Omni-MATH (#3271, #3299, #3271, #3299, #3291, #3322, #3348, #3372, #3373, #3407, #3407)
- IBM enterprise scenarios
    - Add Gold Commodity News scenario (#3065)
    - Add Legal Contract Summarization scenario (#3131)
    - Add CASEHold scenario (#3164)
    - Add SUMO Web Claims Summarization scenario (#3112)
    - Add CTI-to-MITRE scenario (#3249)
    - Add Legal Opinion Sentiment Classification scenario (#3286)
    - Add Financial Phrasebank scenario (#3302)
    - Add ConvFinQACalc (#3453)
    - Add ECHR Judgment Classification scenario (#3311)
- Vision-language model scenarios
    - Fix Image2struct v1.0.1 (#3061)
    - Fix "science & technology" subset of MMSTAR (#3107)
    - Fix R/B channel switch in skin tone calculation (#2589)
- Speech / audio model scenarios
    - Add AudioMNIST scenario (#3093)
    - Add CoVost-2: Speech Machine Translation (#3106)
    - Add Vocal Sound scenario (#3130)
    - Add Multilingual Librispeech (#3130, #3423)
    - Add AudioCaps scenario (#3137)
    - Add IEMOCAP Audio scenario (#3139)
    - Add MELD Audio scenario (#3142)
    - Add FLEURS scenario (#3130, #3151, #3287, #3299)
    - Add Casual Conversation V2 audio scenario (#3158)
    - Add Common_Voice_15 and RSB audio scenarios (#3147)
    - Add Audio PAIRS audio scenario (#3149)
    - Add VoxCeleb2Scenario for audio identification (#3179)
    - Add AIR-Bench chat and foundation audio scenarios (#3189, #3362, #3486)
    - Add MuTox Scenario (#3343)
    - Add MUStARDScenario for sarcasm detection (#3345)
    - Add AMI, LibriSpeech audio scenarios (#3355)
    - Add VoiceJailbreakAttacksScenario (#3361)
    - Add GPT4 evaluator for open-ended audio scenarios (#3417)
    - Add LibriSpeech and FLEURS gender fairness audio scenarios (#3418)
    - Add PARADE audio scenario (#3424)
- Misc
    - Add AutoBencher Capabilities scenario (#3178)
    - Add AutoBencher Safety Scenario (#3165)
    - Add ENEM Challenge Scenario (#3185)
    - Add BigCodeBench (#3186, #3310)
    - Add experimental CzechBankQA scenario (#3222, #3240)
    - Add TweetSentBR Scenario (#3219)
    - Add HarmBench GCG-T (#3035)
    - Add MMLU and Winogrande human-translated into 11 African languages (#3237, #3256)
    - Pin revision in many invocations of Hugging Face load_datasets() (#3124)
    - Add BIRD SQL scenario (#3292, #3312)
    - Add Spider 1.0 scenario (#3300, #3371)
    - Add helpdesk call summarization scenario (#3303, #3387, #3388)
    - Adding IMDB_PTBR Scenario (#3284)
    - Adding OAB Exams Scenario (#3329)
    - Disable test_math_scenario due to DMCA takedown (#3315)
    - Add InfiniteBenchSum scenario (#3409, #3476)
    - Add HotPotQA and SQuAD scenarios from RULER (#3411, #3476)
    - Allow using alternate annotator models for AIR-Bench 2024 (#3468, #3488)
    - Set trust_remote_code for TyDiQA and BANKING77 (#3473, #3477)
- MedHELM scenarios
    - Fix LiveQA scenario (#3244)
    - Add MedHallu scenario (#3483)
    - Add MIMIC-IV-BHC scenario (#3459)
    - Add all conditions in CLEAR scenario (#3466)
    - Add judges to many MedHELM scenarios (#3484)
    - Add or update 31 scenarios for MedHELM v1.0.0 (#3038, #3403, #3492, #3493) -
        ACI-Bench,
        ADHD-Behavior,
        ADHD-MedEffects,
        BMT-Status,
        CDI-QA,
        CLEAR-AD,
        ClinicReferral,
        DischargeMe,
        EHRSHOT,
        EHRSQL,
        ENT-Referral,
        HeadQA,
        HospiceReferral,
        MedAlign,
        Medbullets,
        MedCalc-Bench,
        MedConfInfo,
        MedDialog,
        Medec,
        MedicationQA,
        MEDIQA,
        MentalHealth,
        MIMIC-IV Billing Code,
        MIMIC-RRS,
        MTSamples,
        MTSamples Procedures,
        N2C2-CT Matching,
        NoteExtract,
        PatientInstruct,
        PubMedQA,
        RaceBias

### Models

- Add Mistral and Llama models on AWS Bedrock (#3034, #3092, #3095)
- Add Gemini-1.5-Pro-002 and Gemini-1.5-Flash-002 models (#3032, #3085)
- Add Qwen2.5 Instruct Turbo models on Together AI (#3063)
- Add Anthropic Claude 3.5 Sonnet (20241022) models (#3082)
- Add Mistral Pixtral (2409) (#3073)
- Add GPT-4o Audio Preview (2024-10-01) model (#3091)
- Add Qwen-Audio-Chat and Qwen2-Audio-Instruct (#3104, #3298, #3474)
- Allow setting device for Hugging Face models (#3109)
- Add Mistral Small and Ministral models (#3110)
- Add Llama-Omni-8B (#3119)
- Treat missing AI21 message content as empty string (#3123)
- Add stop sequence support to MistralClient (#3120)
- Deprecate OpenAI legacy completions API (#3144)
- Add Grok Beta model (#3145)
- Add Diva Llama model (#3148)
- Remove OpenVino support (#3153)
- Add support for IBM Granite models on Hugging Face (#3166, #3261)
- Add Claude 3.5 Haiku model (#3171)
- Add Pixtral Large and Mistral Large (2411) models (#3177)
- Add Upstage Solar models (#3181, #3198)
- Add Llama 3.1 Nemotron Instruct (70B) model on Together AI (#3172)
- Add NECTEC model (#3197)
- Add Llama 3.3 model (#3202)
- Add Maritaca AI model (Sabiá 7B) (#3185)
- Add gemini-2.0-flash-exp model (#3210)
- Add Qwen 2 VLM (#3247)
- Add Amazon Nova models (#3251, #3252, #3263, #3264, #3408, #3442)
- Add DeepSeek v3 model (#3253)
- Simplify credential management for Bedrock client (#3255)
- Add Llama 3.1 Instruct on Vertex AI (#3278)
- Add Phi 3.5 models (#3306)
- Add Mistral Small 3 model (#3308)
- Add QwQ model on Together AI (#3307)
- Add Deepseek-R1 model (#3305)
- Add o3-mini model (#3304)
- Handle content filtering from Azure OpenAI (#3319, #3321, #3327)
- Add DeepSeek R1 Distill Llama 8B and DeepSeek Code Instruct 6.7B (#3332)
- Add a version of DeepSeek R1 that hides thinking tokens from output (#3335, #3485)
- Add OpenAI audio models (#3346)
- Add Claude 3.7 model (#3366)
- Add GPT-4.5 model (#3382)
- Added `gemini-2.0-flash-thinking-exp-01-21` (#3410)
- Add SEA-HELM leaderboard and SEA-LIONv3 models (#3347)
- Added `OpenAITranscriptionThenCompletionClient` (#3416)
- Add request response format JSON schema support (#3415)
- Make Azure OpenAI deployment name configurable (#3421)
- Use Anthropic tokenizer from Hugging Face (#3467)
- Add Palmyra Fin model (#3475)
- Add Mistral Small 3.1 model (#3478)
- Added support for phi-3.5 through Azure (#3489)
- Add IBM Granite models hosted on IBM WatsonX (#3441)

### Frontend

- Add web player for audio objects (#3098)
- Add a badge indicating if the release is latest or stale (#3116, #3126)
- Change title for HELM leaderboards (#3117)
- Add functionality for linking directly to instances in Predictions page (#3121)
- Improve leaderboard frontend navigation (#3330)
- Display messages in instances and reuqests in frontend (#3336, #3341)
- Add latest to the frontend version dropdown (#3338)
- Allow overriding Vite base URL with VITE_HELM_FRONTEND_BASE environment variable (#3426, #3428)
- Load project_metadata.json from the website rather than GitHub (#3427)

### Framework

- Add extra_data field to Instance (#3094)
- Fix mean aggregation for tables (#3127, #3309)
- Add encryption for stored data for GPQA (#3216, #3242)
- Allow running recipes from the Unitxt catalog (#3267)
- Add support to redact model outputs (#3301)
- Allow processing output before metrics for reasoning models (#3333)
- Display instances extra_data in frontend (#3340)
- Allow arguments to be passed into annotators (#3487)
- Automatically apply the model=all run expander (#3491)
- Add support for adaptive sampling based on Reliable and Efficient Amortized Model-based Evaluation [(Truong et al., 2025)](https://arxiv.org/abs/2503.13335) (#3397)

### Metrics

- Flip LPIPS so that '1' is better (#3055)
- Fix incorrect handling of labels in ClassificationMetric (#3289)
- Fix ASR WER and MER metrics (#3296)

### Misc

- Change GCS download docs to use gcloud storage instead of gsutil (#3083)
- Fix minor bug in punkt installation logic (#3111)
- Improvements to audio utilities (#3128)
- New arg for quasi-exact match (#3257)
- Use uv for Update Requirements GitHub Action (#3444, #3452)
- Resolve static files using importlib_resources for crfm-proxy-server (#3460, #3461)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @Adam-Kasumovic
- @aptandre
- @arseniy-klimovskiy
- @asad-aali
- @asillycat
- @aunell
- @chiheem
- @farzaank
- @haoqiu1
- @HennyJie
- @ImKeTT
- @JackJessada
- @jessicalundin
- @jmbanda
- @LAOS-Y
- @liamjxu
- @lucas-s-p
- @MiguelAFH
- @Miking98
- @mtake
- @raileymontalan
- @raulista1997
- @rbitr
- @RonalddMatias
- @ryokawajp
- @saikiranjakka
- @sangttruong
- @shakatoday
- @siyagoel
- @subhaviv
- @suhana13
- @teetone
- @thallysonjsa
- @vz-ibm
- @yifanmai
- @YiZeng623
- @yuhengtu

## [v0.5.4] - 2024-10-09

### Breaking Changes

- Python 3.8 is no longer supported - please use Python 3.9 to 3.11 instead.(#2978)

### Scenarios

- Fix prompt for BANKING77 (#3009)
- Split up LINDSEA scenario (#2938)
- Normalize lpips and ssim for image2struct (#3020)

### Models

- Add o1 models (#2989)
- Add Palmyra-X-004 model (#2990)
- Add Palmyra-Med and Palmyra-Fin models (#3028)
- Add Llama 3.2 Turbo models on Together AI (#3029)
- Add Llama 3 Instruct Lite / Turbo on Together AI (#3031)
- Add Llama 3 CPT SEA-Lion v2 models (#3036)
- Add vision support to Together AI client (#3041)

### Frontend

- Display null annotator values correctly in the frontend (#3003)

### Framework

- Add support for Python 3.11 (#2922)
- Fix incorrect handling of ties in win rate computation (#3001, #2008)
- Add mean row aggregation to HELM summarize (#2997, #3030)

### Developer Workflow

- Move pre-commit to pre-push (#3013)
- Improve local frontend pre-commit (#3012)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @brianwgoldman
- @chiheem
- @farzaank
- @JoelNiklaus
- @liamjxu
- @teetone
- @weiqipedia
- @yifanmai

## [v0.5.3] - 2024-09-06

### Breaking Changes

- The `--models-to-run` flag in `helm-run` must now be set if a models run expander such as `models=text` is used (#2852)
- The `--jquery` flag has been removed from `helm-server` because the legacy frontend is no longer supported (#2852)

### Scenarios

- Improve DecodingTrust scenario (#2734, #2600)
- Add BHASA scenarios (#2648, #2914, #2913, #2937)
- Add BHASA LINDSEA scenarios (#2694)
- Change AIR-Bench main score to refusal rate (#2788, #2802, #2873)
- Add EWoK scenario (#2812, #2850, #2882, #2897, #2899)
- Add FinanceBench scenario (#2798)
- Add XSTest Scenario (#2831)
- Add AnthropicRedTeam scenario (#2830)
- Add SimpleSafetyTests Scenario(#2828)
- Add HarmBench Scenario (#2829, #2935)
- Add BANKING77 scenario (#2947)
- Change source dataset URL for Momentos scenario for VHELM (#2823)
- Add RealWorldQA, EXAMS-V, and FairFace scenarios for VHELM (#2825)
- Update Image2Struct scenarios (#2879, #2878, #2888, #2890, #2891, #2919, #2920)

### Models

- Add SambaLingo Thai models (#2747, #2757)
- Add more Typhoon family models (#2745, #2768)
- Add SeaLLM models (#2744)
- Add OpenThaiGPT models (#2743)
- Add SambaLingo-Thai-Base-70B and SambaLingo-Thai-Chat-70B (#2758, #2757, #2782)
- Add Claude 3.5 Sonnet (20240620) (#2763)
- Add multi-GPU support to HuggingFaceClient (#2762)
- Add AI21 Jamba Instruct (#2766)
- Add Gemma 2 and Gemma 2 Instruct models (#2796, #2862)
- Deleted many deprecated models (#2668, #2814)
- Deleted many deprecated window services (#2669)
- Add Phi-3 models (#2815)
- Switched AI21 models to use local tokenizer (#2775)
- Add GPT-4o mini (#2827)
- Add Mistral NeMo (#2826)
- Add Llama 3.1 Instruct Turbo (#2835, #2840, #2844, #2880, #2898)
- Add Mistral Large 2 (#2839)
- Add Nemotron-4-Instruct (#2892, #2896, #2901)
- Add GPT-4o (2024-08-06) (#2900)
- Add Jamba 1.5 models (#2957)
- Add Llama Guard 3 (#2968)

### Frontend

- Fix bug causing repeated renders and excessive CPU usage on some HELM landing pages (#2816)
- Fix bug causing Predictions page to repeatedly download schema.json (#2847)
- Fix spurious AbortError warnings in console logs (#2811)
- Fix incorrect handling perturbations in run predictions frontend (#2950)

### Framework

- Support other reference prefixes in MultipleChoiceJointAdapter (#2809)
- Add validation for --models-to-run (#2852)
- Remove pyext from dependencies (#2921)
- Make Perspective API dependencies optional (#2924)

### Misc

- Add additional instructions for more scenarios in `output_format_instructions` (#2789, #2821, #2822, #2824, #2902, #2906, #2952, #2963)
- Allow the `output_format_instructions` run expander to add additional instructions as suffix (#2964)
- Changelog messages are now in present tense rather than past tense, to align with Git commit message style
- Leaderboard releases are no longer included in this changelog, and will be included in `LEADERBOARD_CHANGELOG.md` instead

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @andyzorigin
- @benediktstroebl
- @danielz02
- @farzaank
- @JosselinSomervilleRoberts
- @percyliang
- @potsawee
- @raileymontalan
- @SathvikNapa
- @shenmishajing
- @teetone
- @weiqipedia
- @yifanmai

## [v0.5.2] - 2024-06-17

### Scenarios

- Updated VHELM scenarios for VLMs (#2719, #2684, #2685, #2641, #2691)
- Updated Image2Struct scenarios (#2608, #2640, #2660, #2661)
- Added Automatic GPT4V Evaluation for VLM Originality Evaluation
- Added FinQA scenario (#2588)
- Added AIR-Bench 2024 (#2698, #2706, #2710, #2712, #2713)
- Fixed `entity_data_imputation` scenario breakage by mirroring source data files (#2750)

### Models

- Added google-cloud-aiplatform~=1.48 dependency requirement for Vertex AI client (#2628)
- Fixed bug with Vertex AI client error handling (#2614)
- Fixed bug with for Arctic tokenizer (#2615)
- Added Qwen1.5 110B Chat (#2621)
- Added TogetherCompletionClient (#2629)
- Fixed bugs with Yi Chat and Llama 3 Chat on Together (#2636)
- Added Optimum Intel (#2609, #2674)
- Added GPT-4o model (#2649, #2656)
- Added SEA-LION 7B and SEA-LION 7B Instruct (#2647) 
- Added more Gemini 1.5 Flash and Pro versions (#2653, #2664, #2718, #2718)
- Added Gemini 1.0 Pro 002 (#2664)
- Added Command R and Command R+ models (#2548) 
- Fixed GPT4V Evaluator Out of Option Range Issue (#2677)
- Added OLMo 1.5 (#2671)
- Added RekaClient (#2675)
- Added PaliGemma (#2683)
- Added Mistral 7B Instruct v0.1, v0.2 and v0.3 (#2665)
- Switched most Together chat models to use the chat client (#2703, #2701, #2705)
- Added MedLM model (#2696, #2709)
- Added Typhoon v1.5 models (#2659)
- Changed HuggingFaceClient to truncate end of text token (#2643)
- Added Qwen2 Instruct (72B) (#2722)
- Added Yi Large (#2723, #1731)
- Added Sailor models (#2658)
- Added BioMistral and Meditron (#2728)

### Frontend

- Miscellaneous improvements and bug fixes (#2618, #2617, #2616, #2651, #2667, #2724)

### Framework

- Removed `adapter_spec` from `schema_*.yaml` files (#2611)
- Added support for annotators / LLM-as-judge (#2622, #2700)
- Updated documentation (#2626, #2529, #2521)

### Evaluation Results

- [MMLU v1.2.0](https://crfm.stanford.edu/helm/mmlu/v1.2.0/)
    - Added results for DBRX Instruct, DeepSeek LLM Chat (67B), Gemini 1.5 Pro (0409 preview), Mistral Small (2402), Mistral Large (2402), Arctic Instruct
- [MMLU v1.3.0](https://crfm.stanford.edu/helm/mmlu/v1.3.0/)
    - Added results for Gemini 1.5 Flash (0514 preview), GPT-4o (2024-05-13), Palmyra X V3 (72B)
- [MMLU v1.4.0](https://crfm.stanford.edu/helm/mmlu/v1.4.0/)
    - Added results for Yi Large (Preview), OLMo 1.7 (7B), Command R, Command R Plus, Gemini 1.5 Flash (001), Gemini 1.5 Pro (001), Mistral Instruct v0.3 (7B), GPT-4 Turbo (2024-04-09), Qwen1.5 Chat (110B), Qwen2 Instruct (72B)
- [Image2Struct v1.0.0](https://crfm.stanford.edu/helm/image2struct/v1.0.0/)
    - Initial release with Claude 3 Sonnet (20240229), Claude 3 Opus (20240229), Gemini 1.0 Pro Vision, Gemini 1.5 Pro (0409 preview),IDEFICS 2 (8B), IDEFICS-instruct (9B), IDEFICS-instruct (80B), LLaVA 1.5 (13B), LLaVA 1.6 (13B), GPT-4o (2024-05-13), GPT-4V (1106 preview), Qwen-VL Chat
- [AIR-Bench v1.0.0](https://crfm.stanford.edu/helm/air-bench/v1.0.0/)
    - Initial release with Claude 3 Haiku (20240307), Claude 3 Sonnet (20240229), Claude 3 Opus (20240229), Cohere Command R, Cohere Command R Plus, DBRX Instruct, DeepSeek LLM Chat (67B), Gemini 1.5 Pro (001, default safety), Gemini 1.5 Flash (001, default safety), Llama 3 Instruct (8B), Llama 3 Instruct (70B), Yi Chat (34B), Mistral Instruct v0.3 (7B), Mixtral Instruct (8x7B), Mixtral Instruct (8x22B), GPT-3.5 Turbo (0613), GPT-3.5 Turbo (1106), GPT-3.5 Turbo (0125), GPT-4 Turbo (2024-04-09), GPT-4o (2024-05-13), Qwen1.5 Chat (72B)

### Contributors

Thank you to the following contributors for your work on this HELM release!

- @andyt-cohere
- @bryanzhou008
- @chiheem
- @farzaank
- @ImKeTT
- @JosselinSomervilleRoberts
- @NoushNabi
- @percyliang
- @raileymontalan
- @shakatoday
- @teetone
- @yifanmai

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

[upcoming]: https://github.com/stanford-crfm/helm/compare/v0.5.11...HEAD
[v0.5.11]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.11
[v0.5.10]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.10
[v0.5.9]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.9
[v0.5.8]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.8
[v0.5.7]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.7
[v0.5.6]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.6
[v0.5.5]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.5
[v0.5.4]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.4
[v0.5.3]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.3
[v0.5.2]: https://github.com/stanford-crfm/helm/releases/tag/v0.5.2
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
