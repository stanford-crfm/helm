# Maintenance Mode Policy

HELM will enter maintenance mode on June 1, 2026. After this date, the following policy will take effect.

HELM will continue to be maintained by volunteer maintainers on a best-effort basis. The HELM code and leaderboards will continue to be available as open-source resources for the community. However, no new features will be added to HELM, and no new evaluations will be added to the HELM leaderboards.

The core HELM software framework is likely to remain functional for an extended period of time. However, many HELM scenarios and models rely on external APIs, which may change in reverse-incompatible ways over time. Additionally, model providers may deprecate models over time. As such, HELM scenarios and models may break due to the lack of active support. When using HELM, it is recommended that you test your scenarios and models to ensure that they are fit for use.

## Reporting Issues

If you find a bug, please open an issue on the Issues page on GitHub. Significant issues may be addressed by maintainers when bandwidth is available.

## User Support

If you have questions about using HELM, please open an issue on the Issues page on GitHub. Maintainers will address questions when bandwidth is available.

## Contributions

The following contributions are welcome:

- Adding notable models and model providers
- Adding notable scenarios
- Updating model API clients to address external breaking API changes
- Fixing significant bugs

A model, model provider or scenario is considered notable if it is widely adopted in industry or academia, or published in a peer-reviewed venue, or of significant interest to a specific community of users.

The following contributions are not welcome:

- Adding new features (especially large or niche features)
- Adding adding non-notable models (especially low-effort finetunes)
- Adding non-notable scenarios that are not notable (especially “homebrew” scenarios)

To contribute, please open an issue on the Issues page on GitHub. The maintainers will reply to acknowledge and triage if the contribution is appropriate. Please wait for the maintainers’ reply, then open a pull request that mentions the issue number in the pull request description. Issues and pull reviews will be reviewed when maintainer bandwidth is available.

All pull requests must have an attached issue. Pull requests without an attached issue will be closed without merging.

## Research Collaborations

You are welcome to use the HELM software or data for your own research. However, the maintainers are no longer able to actively support research collaborations due to the lack of bandwidth.

## Release Cadence

HELM has no fixed release cadence. New versions of the HELM package will be published to PyPI as needed.

## Forking

You may fork the HELM repository for your use cases as long as you comply with the terms of the Apache 2.0 License.

If you intend to maintain a public fork of HELM for community use, please open an issue to discuss this with the maintainers. You should provide proper attribution to HELM as required by our Apache 2.0 License, and you should not claim any endorsement by Stanford University.

## Alternatives

If the lack of active support for HELM is an issue for your use case, you may consider using an alternative open-source LLM evaluation frameworks, such as one of the following (listed in alphabetical order):

- [Evalchemy](https://github.com/mlfoundations/evalchemy)
- [Inspect AI Evals](https://inspect.aisi.org.uk/evals/)
- [Lighteval](https://github.com/huggingface/lighteval)
- [LLM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Unitxt](https://github.com/ibm/unitxt)
