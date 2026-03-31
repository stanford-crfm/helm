---
title: Models
---

# Models

## Introduction

MedHELM model listings are generated from [`model_metadata.yaml`](https://github.com/PacificAI/medhelm/tree/main/src/helm/config/model_metadata.yaml) via the same macros as upstream HELM (`models_by_organization_with_tag`). Deprecated and `simple/*` models are omitted.

## Text Models

{% for tag in ["TEXT_MODEL_TAG", "CODE_MODEL_TAG"] %}
{% for organization, models in models_by_organization_with_tag(tag)|dictsort %}
### {{ organization }}

{% for model in models|sort(attribute="display_name") %}
#### {{ model.display_name }} &mdash; `{{ model.name }}`

{{ model.description }}

{% endfor %}
{% endfor %}
{% endfor %}

## Vision-Language Models

{% for organization, models in models_by_organization_with_tag("VISION_LANGUAGE_MODEL_TAG")|dictsort %}
### {{ organization }}

{% for model in models|sort(attribute="display_name") %}
#### {{ model.display_name }} &mdash; `{{ model.name }}`

{{ model.description }}

{% endfor %}
{% endfor %}

## Text-to-image Models

{% for organization, models in models_by_organization_with_tag("TEXT_TO_IMAGE_MODEL_TAG")|dictsort %}
### {{ organization }}

{% for model in models|sort(attribute="display_name") %}
#### {{ model.display_name }} &mdash; `{{ model.name }}`

{{ model.description }}

{% endfor %}
{% endfor %}

## Audio-Language Models

{% for organization, models in models_by_organization_with_tag("AUDIO_LANGUAGE_MODEL_TAG")|dictsort %}
### {{ organization }}

{% for model in models|sort(attribute="display_name") %}
#### {{ model.display_name }} &mdash; `{{ model.name }}`

{{ model.description }}

{% endfor %}
{% endfor %}
