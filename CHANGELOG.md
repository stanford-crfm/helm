# Changelog

## [Upcoming]

### Models

- Added BigCode (#1506)
- Added GPT-4 (#1457)
- Added OPT 1.3B and 6.7B (#1468)

### Scenarios

- Added legal summarization scenarios (#1454)
- Fixed corner cases in window service truncation (#1449)
- Pinned file order for ICE, APPS (code) and ICE scenarios (#1352)
- Fixed random seed for entity matching scenario (#1475)

### Metrics

- Fixed bug in multi-choice exact match calculation when scores are tied (#1494)

### Framework

- Added script for estimating the cost of a run suite (#1480)
- Added support for human critique evaluation using Surge AI (#1330)

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

[upcoming]: https://github.com/stanford-crfm/helm/compare/v0.2.2...HEAD
[v0.2.2]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.2
[v0.2.1]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.1
[v0.2.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.0
[v0.1.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.1.0
