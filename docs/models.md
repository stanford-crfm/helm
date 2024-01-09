# Models

{% for model_type, model_tag in model_types_and_tags().items() %}

## {{ model_type }} Models

{% for organization, models in models_by_organization_with_tag(model_tag).items() %}

### {{ organization }}

{% for model in models %}

#### {{ model.display_name }} (`{{ model.name }}`)

{{ model.description }}

{% endfor %}
{% endfor %}
{% endfor %}

## HEIM (text-to-image evaluation)

For a list of image-generation models, please visit the [models page](https://crfm.stanford.edu/heim/latest/?models) of the HEIM results website.
