# Changelog

## [Upcoming]

- Add changes for the upcoming release here

## [v0.2.1]

### Models

- Added BigCode SantaCoder (#1312)
- Added OPT-IML, Cohere Command, CodeGeeX (#1331)

### Scenarios

- Added LEXTREME and LexGLUE legal scenarios (#1216)
- Added WMT14 machine translation scenario (#1329)
- Added biomedical scenarios: COVID Dialogue, MeQSum, MedDialog, MedMCQA, MedParagraphSimplification, MedQA, PubMedQA (#1332)

### Framework

- Added `--run-specs` flag to `helm-run` (#1302)
- Reduced running time of `helm-summarize` (#1269)
- Added classification metrics (#1368)

## [v0.2.0]

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

[upcoming]: https://github.com/stanford-crfm/helm/compare/v0.2.1...HEAD
[v0.2.1]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.1
[v0.2.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.2.0
[v0.1.0]: https://github.com/stanford-crfm/helm/releases/tag/v0.1.0
