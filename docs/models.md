# Models

## Text Models

{% for tag in ["TEXT_MODEL_TAG", "CODE_MODEL_TAG"] %}

{% for organization, models in models_by_organization_with_tag(tag).items() %}

### {{ organization }}

{% for model in models %}

#### {{ model.display_name }} &mdash; `{{ model.name }}`

{{ model.description }}

{% endfor %}
{% endfor %}
{% endfor %}

## HEIM (text-to-image evaluation)

For a list of text-to-image models, please visit the [models page](https://crfm.stanford.edu/heim/latest/?models) of the HEIM results website.
